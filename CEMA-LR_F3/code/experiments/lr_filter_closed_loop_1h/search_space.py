from __future__ import annotations

from functools import lru_cache

from experiments.lr_cross_backbone_validation_1h.protocol import generate_candidate_pools as generate_extra
from experiments.lr_gru_tcn_paired_1h.protocol import generate_candidate_pools as generate_core


BACKBONES = ("gru", "tcn", "lstm", "bigru", "transformer")


@lru_cache(maxsize=1)
def candidate_pools() -> dict:
    """Return the frozen five-candidate pools used by the two historical loops."""
    core = generate_core(42)
    extra = generate_extra(42)
    return {name: (core if name in {"gru", "tcn"} else extra)[name] for name in BACKBONES}


def strip_metadata(config: dict) -> dict:
    return {
        key: value
        for key, value in config.items()
        if key not in {"candidate_id", "dilations", "receptive_field"}
    }

