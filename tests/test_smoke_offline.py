"""Run the offline smoke checks in the default gate; browser checks stay opt-in."""

import importlib.util
import re
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts/smoke_offline.py"
FEATURE_MAP = ROOT / "docs/verification.md"


def load_smoke(monkeypatch):
    spec = importlib.util.spec_from_file_location("smoke_offline", SCRIPT)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    # dataclasses resolve the module through sys.modules while the script loads.
    monkeypatch.setitem(sys.modules, spec.name, module)
    spec.loader.exec_module(module)
    return module


def test_offline_smoke_checks_pass_and_match_the_feature_map(monkeypatch, tmp_path):
    smoke = load_smoke(monkeypatch)
    config = smoke.build_offline_index(tmp_path / "smoke")
    checks = smoke.run_pipeline_checks(config) + smoke.run_ui_checks(config)

    assert [f"{c.feature} {c.name}: {c.detail}" for c in checks if not c.passed] == []
    rows = re.findall(r"^\| (F\d+) \|(.*)\|$", FEATURE_MAP.read_text(encoding="utf-8"), re.MULTILINE)
    assert {c.feature for c in checks} == {feature for feature, row in rows if "smoke:" in row}
