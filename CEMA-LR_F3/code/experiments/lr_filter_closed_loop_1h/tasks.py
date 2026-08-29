from __future__ import annotations

from dataclasses import asdict, dataclass

from .protocol import FORMAL_SEEDS
from .search_space import BACKBONES, candidate_pools


DATASETS = ("FC1", "FC2")
TARGET_MODES = ("direct", "lr")
FOLDS = (1, 2, 3)


@dataclass(frozen=True)
class FitTask:
    stage: str
    filter_id: str
    dataset: str
    backbone: str
    target_mode: str
    replicate: str
    config_id: str

    @property
    def task_id(self) -> str:
        return "__".join((self.stage, self.filter_id, self.dataset, self.backbone, self.target_mode, self.config_id, self.replicate))

    def to_dict(self) -> dict:
        return asdict(self) | {"task_id": self.task_id}


def build_shared_search_tasks(filter_id: str) -> list[FitTask]:
    output = []
    pools = candidate_pools()
    for dataset in DATASETS:
        for backbone in BACKBONES:
            for config in pools[backbone]["primary"]:
                for target_mode in TARGET_MODES:
                    for fold in FOLDS:
                        output.append(FitTask("shared", filter_id, dataset, backbone, target_mode, f"fold{fold}", config["candidate_id"]))
    return output


def build_independent_tuning_tasks(filter_id: str, trials: int = 10) -> list[FitTask]:
    return [
        FitTask("tune", filter_id, dataset, backbone, target_mode, f"fold{fold}", f"trial{trial:02d}")
        for dataset in DATASETS
        for backbone in BACKBONES
        for target_mode in TARGET_MODES
        for trial in range(trials)
        for fold in FOLDS
    ]


def build_formal_tasks(filter_id: str) -> list[FitTask]:
    return [
        FitTask("formal", filter_id, dataset, backbone, target_mode, f"seed{seed}", "selected")
        for dataset in DATASETS
        for backbone in BACKBONES
        for target_mode in TARGET_MODES
        for seed in FORMAL_SEEDS
    ]

