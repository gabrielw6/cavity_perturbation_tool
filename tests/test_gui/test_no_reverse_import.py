"""docs/gui_module_plan.md Section 1.1/8/10 -- the one-way-dependency
guard, from day one: `cavity_perturbation` must never import, reference, or
know about `cavity_perturbation_gui`. Checked mechanically (a grep), not by
convention alone."""
import re
from pathlib import Path

_SRC_ROOT = Path(__file__).resolve().parents[2] / "src" / "cavity_perturbation"
_FORBIDDEN = re.compile(r"cavity_perturbation_gui")


def test_cavity_perturbation_never_references_the_gui_package():
    assert _SRC_ROOT.is_dir(), f"expected {_SRC_ROOT} to exist"
    offending: list[str] = []
    for path in _SRC_ROOT.rglob("*.py"):
        text = path.read_text(encoding="utf-8")
        if _FORBIDDEN.search(text):
            offending.append(str(path))
    assert offending == [], f"cavity_perturbation must never reference cavity_perturbation_gui, found in: {offending}"
