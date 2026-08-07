"""Package metadata smoke tests."""

from __future__ import annotations

import care_asd


def test_version_string() -> None:
    assert isinstance(care_asd.__version__, str)
    assert care_asd.__version__.count(".") >= 1
