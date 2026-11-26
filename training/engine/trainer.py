"""Training/validation loops and experiment lifecycle."""
from pathlib import Path
import logging
import sys
import time
import uuid
import numpy as np
import torch
from training.models import build_model, forward_batch
from training.data.loaders import make_loader
from training.data.preprocessing import load_metadata
from training.settings import write_json
from .runtime import select_device, seed_all, rng_snapshot, restore_rng
from .checkpoints import atomic_save, training_contract, load_model_weights
from .history import save_history

LOGGER = logging.getLogger(__name__)

def run_epoch(model, loader, device, optimizer=None, scheduler=None, log_every=5):
    training = optimizer is not None
    model.train(training)
    total, elements = 0.0, 0
    with torch.set_grad_enabled(training):
        for step, (frames, labels, _, _) in enumerate(loader, 1):
            frames, labels = frames.to(device, non_blocking=True), labels.to(device, non_blocking=True)
            if training:
                optimizer.zero_grad(set_to_none=True)
            predicted = forward_batch(model, frames)
            loss = torch.nn.functional.mse_loss(predicted, labels)
            if not torch.isfinite(loss):
                raise RuntimeError("Non-finite loss; inspect input data and learning rate")
            if training:
                loss.backward()
                optimizer.step()
                if scheduler is not None:
                    scheduler.step()
            total += loss.item() * labels.numel()
            elements += labels.numel()
            if step % log_every == 0 or step == len(loader):
                LOGGER.info("%s step %d/%d | loss %.6f", "train" if training else "valid", step, len(loader), loss.item())
    return total / elements

def train(cfg, args):
    seed_all(cfg["train"]["seed"])
    generator = torch.Generator().manual_seed(cfg["train"]["seed"])
    train_loader = make_loader(cfg, "train", generator)
    valid_loader = make_loader(cfg, "valid")
    device = select_device(cfg)
    model = build_model(cfg).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg["train"]["learning_rate"], weight_decay=cfg["train"]["weight_decay"])
    scheduler = None
    if cfg["train"].get("scheduler", "onecycle") == "onecycle":
        scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer, max_lr=cfg["train"]["learning_rate"],
                            epochs=cfg["train"]["epochs"], steps_per_epoch=len(train_loader))
    start, best, history = 0, float("inf"), []
    if args.resume:
        checkpoint = load_model_weights(model, args.resume)
        if checkpoint.get("contract") != training_contract(cfg) or checkpoint.get("steps_per_epoch") != len(train_loader):
            raise ValueError("Resume configuration differs. For a new schedule/data split use --init-weights instead.")
        expected_sources = {name: load_metadata(cfg, name)[1]["source"] for name in cfg["subjects"]["train"] + cfg["subjects"]["valid"]}
        if checkpoint["sources"] != expected_sources:
            raise ValueError("Resume data differs from checkpoint")
        optimizer.load_state_dict(checkpoint["optimizer"])
        if scheduler is not None:
            scheduler.load_state_dict(checkpoint["scheduler"])
        elif checkpoint["scheduler"] is not None:
            raise ValueError("Resume scheduler does not match constant learning rate")
        start, best, history = checkpoint["epoch"], checkpoint["best_valid_loss"], checkpoint["history"]
        restore_rng(checkpoint["rng"], generator)
        run_dir = Path(args.resume).resolve().parent
    else:
        if args.init_weights:
            load_model_weights(model, args.init_weights)
        run_dir = Path(cfg["paths"]["runs_dir"]) / cfg["model"].get("name", "efficientphys_c") / (time.strftime("%Y%m%d_%H%M%S") + "_" + uuid.uuid4().hex[:6])
        run_dir.mkdir(parents=True, exist_ok=False)
        write_json(run_dir / "config.json", cfg)
        write_json(run_dir / "environment.json", {"python": sys.version, "torch": torch.__version__,
                   "numpy": np.__version__, "cuda": torch.version.cuda, "device": str(device),
                   "gpu": torch.cuda.get_device_name(device) if device.type == "cuda" else None,
                   "init_weights": str(Path(args.init_weights).resolve()) if args.init_weights else None})
    file_handler = logging.FileHandler(run_dir / "train.log", encoding="utf-8")
    file_handler.setFormatter(logging.Formatter("%(asctime)s [%(levelname)s] %(message)s"))
    logging.getLogger().addHandler(file_handler)
    end = cfg["train"]["epochs"] if args.stop_after_epoch is None else args.stop_after_epoch
    if not start < end <= cfg["train"]["epochs"]:
        raise ValueError("stop-after-epoch must be greater than completed epochs and <= configured total epochs")
    LOGGER.info("Run %s | model %s | device %s | loss mse | scheduler %s | train %d clips | valid %d clips",
                run_dir, cfg["model"].get("name", "efficientphys_c"), device,
                cfg["train"].get("scheduler", "onecycle"), len(train_loader.dataset), len(valid_loader.dataset))
    try:
        for epoch in range(start, end):
            began = time.perf_counter()
            train_loss = run_epoch(model, train_loader, device, optimizer, scheduler, cfg["train"]["log_every"])
            valid_loss = run_epoch(model, valid_loader, device, log_every=cfg["train"]["log_every"])
            improved = valid_loss < best
            best = min(best, valid_loss)
            history.append({"epoch": epoch + 1, "train_loss": train_loss, "valid_loss": valid_loss,
                            "learning_rate": optimizer.param_groups[0]["lr"], "seconds": time.perf_counter() - began})
            checkpoint = {"model": model.state_dict(), "optimizer": optimizer.state_dict(),
                "scheduler": scheduler.state_dict() if scheduler is not None else None, "epoch": epoch + 1, "best_valid_loss": best,
                "history": history, "config": cfg, "contract": training_contract(cfg),
                "steps_per_epoch": len(train_loader), "rng": rng_snapshot(generator),
                "sources": {name: load_metadata(cfg, name)[1]["source"] for name in cfg["subjects"]["train"] + cfg["subjects"]["valid"]}}
            atomic_save(run_dir / f"epoch_{epoch + 1:03d}.pt", checkpoint)
            atomic_save(run_dir / "last.pt", checkpoint)
            if improved:
                atomic_save(run_dir / "best.pt", checkpoint)
                atomic_save(run_dir / "best_weights.pth", model.state_dict())
                write_json(run_dir / "best_weights.metadata.json", {"model": cfg["model"], "preprocess": cfg["preprocess"], "epoch": epoch + 1})
            save_history(run_dir, history, cfg["model"].get("name", "efficientphys_c"))
            LOGGER.info("Epoch %d/%d | train %.6f | valid %.6f | best %.6f | checkpoint saved", epoch + 1, cfg["train"]["epochs"], train_loss, valid_loss, best)
    except KeyboardInterrupt:
        LOGGER.warning("Interrupted; resume from last.pt at the last completed epoch (partial epoch is not saved)")
    finally:
        logging.getLogger().removeHandler(file_handler)
        file_handler.close()
