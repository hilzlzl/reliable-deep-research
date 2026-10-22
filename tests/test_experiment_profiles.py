"""Regression tests for baseline and reliable execution paths."""

from open_deep_research.configuration import Configuration


def test_reliable_runtime_can_be_disabled_explicitly():
    baseline = Configuration.from_runnable_config(
        {"configurable": {"enable_reliable_runtime": False}}
    )
    reliable = Configuration.from_runnable_config(
        {"configurable": {"enable_reliable_runtime": True}}
    )

    assert baseline.enable_reliable_runtime is False
    assert reliable.enable_reliable_runtime is True
