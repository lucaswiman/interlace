"""Add the tests directory to sys.path so case_study_helpers can be imported."""

import os
import sys

_test_dir = os.path.dirname(os.path.abspath(__file__))
if _test_dir not in sys.path:
    sys.path.insert(0, _test_dir)
