#!/usr/bin/env python3
"""RLT stage one: train the RL-token encoder-decoder on a frozen VLA's prefix features.

The VLA (pi0/pi05 PyTorch) SFT checkpoint is loaded frozen; each step runs a
prefix-only forward (`PI0Pytorch.extract_prefix_hidden`) and trains the
RLTTokenTransformer to compress the prefix hidden states into RL tokens and
autoregressively reconstruct them (masked MSE). The VLA is never updated.

If the SFT checkpoint is JAX/orbax, convert it first with
examples/convert_jax_model_to_pytorch.py - the feature-extraction backend used
here must match the backend later used for stage-two inference.

Usage:
    python scripts/rlt/train_token.py --config rlt_test --exp-name run01
"""

import argparse
import dataclasses
import logging
import pathlib
import shutil
import time

import jax
import numpy as np
import safetensors.torch
import torch
import tqdm

import openpi.models.pi0_config as pi0_config
import openpi.models_pytorch.pi0_pytorch as pi0_pytorch
import openpi.rlt.config as rlt_config
from openpi.rlt.token_model import RLTTokenTransformer
import openpi.training.config as _config
import openpi.training.data_loader as _data


def build_token_model(model_cfg: rlt_config.RLTModelConfig) -> RLTTokenTransformer:
    return RLTTokenTransformer(
        input_dim=model_cfg.input_dim,
        embed_dim=model_cfg.d_model,
        prefix_seq_len=model_cfg.prefix_seq_len,
        num_rl_tokens=model_cfg.num_rl_tokens,
        num_layers=model_cfg.num_encoder_layers,
        num_decoder_layers=model_cfg.num_decoder_layers,
        num_heads=model_cfg.num_heads,
        mlp_ratio=model_cfg.dim_feedforward / model_cfg.d_model,
        dropout_rate=model_cfg.dropout,
    )


def save_checkpoint(
    token_model: RLTTokenTransformer,
    optimizer: torch.optim.Optimizer,
    global_step: int,
    config: rlt_config.RLTConfig,
) -> None:
    """Atomically save token-model + optimizer state, then enforce keep_period."""
    ckpt_root = config.token_checkpoint_dir
    final_dir = ckpt_root / f"{global_step}"
    tmp_dir = ckpt_root / f"tmp_{global_step}"

    if tmp_dir.exists():
        shutil.rmtree(tmp_dir)
    tmp_dir.mkdir(parents=True, exist_ok=True)

    safetensors.torch.save_model(token_model, tmp_dir / "model.safetensors")
    torch.save(optimizer.state_dict(), tmp_dir / "optimizer.pt")
    torch.save(
        {"global_step": global_step, "config": dataclasses.asdict(config), "timestamp": time.time()},
        tmp_dir / "metadata.pt",
    )

    if final_dir.exists():
        shutil.rmtree(final_dir)
    tmp_dir.rename(final_dir)
    logging.info(f"Saved checkpoint at step {global_step} -> {final_dir}")

    # Keep the latest checkpoint plus any step matching keep_period (orbax-style
    # max_to_keep=1 semantics; the PyTorch training path has no such cleanup).
    step_dirs = sorted(
        (d for d in ckpt_root.iterdir() if d.is_dir() and d.name.isdigit()),
        key=lambda d: int(d.name),
    )
    for d in step_dirs[:-1]:
        if config.keep_period is None or int(d.name) % config.keep_period != 0:
            shutil.rmtree(d)


def load_checkpoint(
    token_model: RLTTokenTransformer,
    optimizer: torch.optim.Optimizer,
    config: rlt_config.RLTConfig,
    device: torch.device,
) -> int:
    """Load the latest token checkpoint and return its step."""
    ckpt_root = config.token_checkpoint_dir
    steps = [int(d.name) for d in ckpt_root.iterdir() if d.is_dir() and d.name.isdigit()] if ckpt_root.exists() else []
    if not steps:
        raise FileNotFoundError(f"No checkpoints found in {ckpt_root} for resume")
    latest = max(steps)
    ckpt_dir = ckpt_root / f"{latest}"
    safetensors.torch.load_model(token_model, ckpt_dir / "model.safetensors", device=str(device))
    optimizer_path = ckpt_dir / "optimizer.pt"
    if optimizer_path.exists():
        optimizer.load_state_dict(torch.load(optimizer_path, map_location=device, weights_only=False))
    logging.info(f"Resumed token checkpoint from step {latest}")
    return latest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, help="RLT config name (configs/rlt/<name>.yaml)")
    parser.add_argument("--exp-name", required=True, help="Experiment name; names the checkpoint directory")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    config = dataclasses.replace(rlt_config.get_config(args.config), exp_name=args.exp_name)
    tt_cfg = config.token_training

    torch.manual_seed(config.seed)
    np.random.seed(config.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Checkpoint directory semantics (mirrors scripts/train_pytorch.py).
    resuming = False
    if config.resume:
        resuming = True  # load_checkpoint below fails loudly if nothing to resume
    elif config.overwrite and config.checkpoint_dir.exists():
        shutil.rmtree(config.checkpoint_dir)
        logging.info(f"Overwriting checkpoint directory: {config.checkpoint_dir}")
    config.token_checkpoint_dir.mkdir(parents=True, exist_ok=True)

    wandb_run = None
    if config.wandb_enabled:
        import wandb

        wandb_run = wandb.init(
            project=config.project_name,
            name=f"{config.name}/{config.exp_name}",
            config=dataclasses.asdict(config),
            resume="allow" if resuming else None,
        )

    # Frozen VLA feature extractor. The TrainConfig supplies the architecture and
    # data pipeline; the RLT config's repo_id overrides the dataset.
    train_cfg = _config.get_config(tt_cfg.vla_config)
    if not isinstance(train_cfg.model, pi0_config.Pi0Config):
        raise ValueError(f"vla_config '{tt_cfg.vla_config}' must use a Pi0Config model, got {type(train_cfg.model)}")
    train_cfg = dataclasses.replace(
        train_cfg,
        data=dataclasses.replace(train_cfg.data, repo_id=tt_cfg.repo_id),
    )
    object.__setattr__(train_cfg.model, "dtype", train_cfg.pytorch_training_precision)

    vla = pi0_pytorch.PI0Pytorch(train_cfg.model).to(device)
    vla_weights = pathlib.Path(tt_cfg.vla_checkpoint) / "model.safetensors"
    if not vla_weights.is_file():
        raise FileNotFoundError(
            f"VLA checkpoint not found: {vla_weights}. JAX/orbax checkpoints must be converted first "
            "(examples/convert_jax_model_to_pytorch.py)."
        )
    safetensors.torch.load_model(vla, vla_weights)
    vla.eval()
    for param in vla.parameters():
        param.requires_grad = False
    logging.info(f"Loaded frozen VLA from {vla_weights}")

    # Prefix features depend only on images + prompt, so action/state norm stats
    # are unnecessary.
    data_config = train_cfg.data.create(train_cfg.assets_dirs, train_cfg.model)
    loader = _data.create_torch_data_loader(
        data_config,
        model_config=train_cfg.model,
        action_horizon=train_cfg.model.action_horizon,
        batch_size=tt_cfg.batch_size,
        shuffle=True,
        num_workers=tt_cfg.num_workers,
        seed=config.seed,
        skip_norm_stats=True,
        framework="pytorch",
    )

    token_model = build_token_model(config.model).to(device)
    optimizer = torch.optim.AdamW(token_model.parameters(), lr=tt_cfg.learning_rate)

    global_step = load_checkpoint(token_model, optimizer, config, device) if resuming else 0

    token_model.train()
    logging.info(
        f"Training RLT token model: steps={tt_cfg.num_train_steps} batch_size={tt_cfg.batch_size} "
        f"lr={tt_cfg.learning_rate} z_dim={token_model.z_dim} params={sum(p.numel() for p in token_model.parameters())}"
    )

    infos = []
    start_time = time.time()
    pbar = tqdm.tqdm(total=tt_cfg.num_train_steps, initial=global_step, desc="Token training")
    while global_step < tt_cfg.num_train_steps:
        for observation, _actions in loader:
            if global_step >= tt_cfg.num_train_steps:
                break
            observation = jax.tree.map(lambda x: x.to(device), observation)

            with torch.no_grad():
                prefix_hidden, prefix_mask = vla.extract_prefix_hidden(observation, image_only=config.model.image_only)

            mask = prefix_mask if config.model.use_mask else None
            loss, _info = token_model(prefix_hidden.detach(), mask)

            loss.backward()
            grad_norm = torch.nn.utils.clip_grad_norm_(token_model.parameters(), max_norm=1.0)
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)

            global_step += 1
            pbar.update(1)
            infos.append({"loss": loss.item(), "grad_norm": float(grad_norm)})

            if global_step % config.log_interval == 0:
                avg_loss = sum(i["loss"] for i in infos) / len(infos)
                avg_grad = sum(i["grad_norm"] for i in infos) / len(infos)
                elapsed = time.time() - start_time
                logging.info(
                    f"step={global_step} loss={avg_loss:.4f} grad_norm={avg_grad:.2f} time={elapsed:.1f}s"
                )
                if wandb_run is not None:
                    wandb_run.log(
                        {
                            "loss": avg_loss,
                            "grad_norm": avg_grad,
                            "time_per_step": elapsed / config.log_interval,
                        },
                        step=global_step,
                    )
                infos = []
                start_time = time.time()

            if global_step % config.save_interval == 0 or global_step == tt_cfg.num_train_steps:
                save_checkpoint(token_model, optimizer, global_step, config)

    pbar.close()
    if wandb_run is not None:
        wandb_run.finish()


if __name__ == "__main__":
    main()
