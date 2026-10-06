import copy
from importlib import resources

import pytest
import yaml

from fertigung.shop.config import load_shop
from fertigung.shop.validation import validate_shop

WORKSHOP = yaml.safe_load(
    resources.files("fertigung.configs").joinpath("workshop.yaml").read_text(encoding="utf-8")
)


def test_workshop_is_valid():
    config = load_shop(WORKSHOP)
    assert validate_shop(config) == []
    assert config.objective.resolved()["lateness"] == 50


def _broken(change):
    data = copy.deepcopy(WORKSHOP)
    change(data)
    with pytest.raises(ValueError) as e:
        load_shop(data)
    return str(e.value)


def test_structural_errors():
    second_raw = {"name": "second-raw", "kind": "raw", "position": [5, 5]}
    assert "exactly one raw store" in _broken(lambda d: d["layout"]["stores"].append(second_raw))
    unknown = [{"from": "thin", "to": "unknown", "minutes": 5}]
    assert "setup family 'unknown'" in _broken(lambda d: d["machine_types"][0]["setup"].update(times=unknown))
    assert "end must be after start" in _broken(lambda d: d["staff"]["shifts"][0].update(end="05:00"))


def test_semantic_errors():
    data = copy.deepcopy(WORKSHOP)
    data["machines"][3]["input_buffer"] = 3  # welding needs five inputs for a frame
    data["machine_types"][3]["setup"]["operators"] = 7
    data["transformations"][5]["duration"] = 1000  # coating may not be interrupted, staffed stretch is 960
    messages = [i.message for i in validate_shop(load_shop(data)) if i.level == "error"]
    assert any("input buffer 3 is smaller than the 5 inputs of 'weld-frame'" in m for m in messages)
    assert any("setting up machine 'coating-1' needs 7 workers" in m for m in messages)
    assert any(
        "'coat-frame' on 'coating-line' takes 1000 minutes without interruption" in m for m in messages
    )
