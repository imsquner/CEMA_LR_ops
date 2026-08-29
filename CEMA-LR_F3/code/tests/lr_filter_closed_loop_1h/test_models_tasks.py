import copy

import pytest
import torch

from experiments.lr_filter_closed_loop_1h.models import build_model, paired_initial_states, state_hash
from experiments.lr_filter_closed_loop_1h.search_space import candidate_pools, strip_metadata
from experiments.lr_filter_closed_loop_1h.tasks import (
    build_formal_tasks,
    build_independent_tuning_tasks,
    build_shared_search_tasks,
)


BACKBONES = ("gru", "tcn", "lstm", "bigru", "transformer")


@pytest.mark.parametrize("backbone", BACKBONES)
def test_all_backbones_accept_m5_windows(backbone):
    config = strip_metadata(candidate_pools()[backbone]["primary"][0])
    model = build_model(backbone, config)
    assert model(torch.zeros(2, 12, 5)).shape == (2, 1)


@pytest.mark.parametrize("backbone", BACKBONES)
def test_direct_lr_pair_has_identical_but_independent_initial_state(backbone):
    config = strip_metadata(candidate_pools()[backbone]["primary"][0])
    left, right = paired_initial_states(backbone, config, seed=42)
    assert state_hash(left) == state_hash(right)
    first_key = next(iter(left))
    right[first_key] = right[first_key].clone()
    right[first_key].view(-1)[0] += 1
    assert state_hash(left) != state_hash(right)


def test_shared_search_inventory_is_exactly_300_fold_fits_per_filter():
    tasks = build_shared_search_tasks("F1")
    assert len(tasks) == 300
    assert len({task.task_id for task in tasks}) == 300
    assert {task.filter_id for task in tasks} == {"F1"}


def test_independent_tuning_inventory_is_600_fold_fits_per_filter():
    tasks = build_independent_tuning_tasks("F1", trials=10)
    assert len(tasks) == 600
    assert len({task.task_id for task in tasks}) == 600


def test_formal_inventory_is_100_models_per_filter():
    tasks = build_formal_tasks("F1")
    assert len(tasks) == 100
    assert len({task.task_id for task in tasks}) == 100

