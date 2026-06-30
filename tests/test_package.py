"""Smoke test — verifies package imports after Phase 1."""

import src


def test_package_version():
    assert src.__version__ == "0.1.0"
