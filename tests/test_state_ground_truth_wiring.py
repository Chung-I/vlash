"""use_state_ground_truth must flow from TrainConfig into both dataset classes."""
import inspect
from unittest.mock import patch

from vlash.configs.train_config import VLASHTrainConfig
import vlash.train as train_mod


def test_config_has_field():
    assert "use_state_ground_truth" in {f.name for f in VLASHTrainConfig.__dataclass_fields__.values()}
    assert VLASHTrainConfig.__dataclass_fields__["use_state_ground_truth"].default is False


def test_make_dataset_passes_flag_shared():
    src = inspect.getsource(train_mod)
    # Both constructor call sites must forward the flag
    assert src.count("use_state_ground_truth=cfg.use_state_ground_truth") == 2
