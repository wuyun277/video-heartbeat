"""Export plain weights with their architecture and preprocessing metadata."""
from training.models import build_model
from training.settings import cache_signature, model_contract, write_json
from .checkpoints import atomic_save, load_model_weights
import logging

LOGGER = logging.getLogger(__name__)


def export_weights(cfg, args):
    model = build_model(cfg)
    checkpoint = load_model_weights(model, args.checkpoint)
    if "config" in checkpoint and (
        model_contract(checkpoint["config"]) != model_contract(cfg)
        or cache_signature(checkpoint["config"]) != cache_signature(cfg)
    ):
        raise ValueError("Export configuration does not match training checkpoint")
    if args.output.exists():
        raise FileExistsError(args.output)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    atomic_save(args.output, model.state_dict())
    write_json(args.output.with_suffix(".metadata.json"), {
        "model": cfg["model"], "preprocess": cfg["preprocess"]})
    LOGGER.info("Exported %s", args.output)
