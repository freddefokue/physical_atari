from __future__ import annotations

from argparse import Namespace

import pytest

from benchmark.run_single_game import validate_args


def test_validate_args_carmack_requires_frame_skip_1():
    args = Namespace(runner_mode="carmack_compat", frame_skip=4)
    with pytest.raises(ValueError, match=r"carmack_compat requires --frame-skip 1"):
        validate_args(args)


def test_validate_args_standard_allows_frame_skip_not_one():
    args = Namespace(runner_mode="standard", frame_skip=4)
    validate_args(args)

