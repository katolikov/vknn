"""pytest bootstrap: make `onnx_optimizer` importable from anywhere.

The package lives under tools/ (not installed); tests may be launched from
the repo root, from tools/, or from this directory.
"""
import os
import sys

_TOOLS_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if _TOOLS_DIR not in sys.path:
    sys.path.insert(0, _TOOLS_DIR)
