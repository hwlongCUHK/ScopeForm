"""Locate the unchanged collaborator package beside this experiment."""
import os
from pathlib import Path
import sys

PACKAGE_ROOT = Path(os.environ.get(
    "CPM_PACKAGE_ROOT",
    str(Path(__file__).resolve().parent.parent / "CPM_Sandbox_collaborator_complete_20261112"),
)).resolve()
RUNTIME = PACKAGE_ROOT / "medical_scale_v06"
sys.path.insert(0, str(RUNTIME))

