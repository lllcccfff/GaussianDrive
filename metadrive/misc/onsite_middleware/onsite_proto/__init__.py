import os
import sys

# Make generated top-level proto packages (main/, chassis/) importable.
_THIS_DIR = os.path.dirname(__file__)
if _THIS_DIR not in sys.path:
    sys.path.insert(0, _THIS_DIR)
