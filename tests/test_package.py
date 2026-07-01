"""Smoke test — verifies package imports after restructure."""

import backend


def test_package_version():
    assert backend.__version__ == "0.1.0"
