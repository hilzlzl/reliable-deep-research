"""Configuration precedence tests."""

import pytest
from pydantic import ValidationError

from open_deep_research.configuration import Configuration, SearchAPI
from open_deep_research.utils import get_api_key_for_model


def test_runtime_configuration_overrides_environment(monkeypatch):
    monkeypatch.setenv("RESEARCH_MODEL", "openai:environment-model")

    configuration = Configuration.from_runnable_config(
        {"configurable": {"research_model": "openai:runtime-model"}}
    )

    assert configuration.research_model == "openai:runtime-model"


def test_environment_is_used_as_fallback(monkeypatch):
    monkeypatch.setenv("RESEARCH_MODEL", "openai:environment-model")

    configuration = Configuration.from_runnable_config({"configurable": {}})

    assert configuration.research_model == "openai:environment-model"


def test_model_api_key_can_use_dedicated_environment_variable(monkeypatch):
    monkeypatch.setenv("DEEPSEEK_API_KEY", "test-secret")
    config = {"configurable": {"model_api_key_env": "DEEPSEEK_API_KEY"}}

    assert get_api_key_for_model("openai:deepseek-flash", config) == "test-secret"


def test_total_research_unit_budget_is_configurable_and_positive():
    configuration = Configuration.from_runnable_config(
        {"configurable": {"max_total_research_units": 2}}
    )

    assert configuration.max_total_research_units == 2
    with pytest.raises(ValidationError):
        Configuration(max_total_research_units=0)


def test_structured_report_controls_are_opt_in():
    configuration = Configuration()

    assert configuration.enable_structured_report is False
    assert configuration.enable_gap_search is False
    assert configuration.max_initial_research_units is None
    assert configuration.enable_adaptive_research is False
    assert configuration.enable_sentence_audit is False


def test_adaptive_research_limits_are_validated():
    configuration = Configuration(
        enable_adaptive_research=True,
        max_adaptive_rounds=3,
        max_adaptive_actions_per_round=2,
    )

    assert configuration.max_adaptive_rounds == 3
    with pytest.raises(ValidationError):
        Configuration(max_adaptive_rounds=0)
    with pytest.raises(ValidationError):
        Configuration(adaptive_budget_reserve_fraction=1.0)


def test_research_task_timeout_is_configurable_and_positive():
    configuration = Configuration.from_runnable_config(
        {"configurable": {"research_task_timeout_seconds": 240}}
    )

    assert configuration.research_task_timeout_seconds == 240
    with pytest.raises(ValidationError):
        Configuration(research_task_timeout_seconds=0)


def test_adaptive_hybrid_search_configuration():
    configuration = Configuration.from_runnable_config(
        {
            "configurable": {
                "search_api": "hybrid",
                "hybrid_local_min_results": 4,
                "hybrid_local_min_unique_documents": 3,
                "hybrid_local_min_query_term_coverage": 0.8,
            }
        }
    )

    assert configuration.search_api == SearchAPI.HYBRID
    assert configuration.hybrid_local_min_results == 4
    assert configuration.hybrid_local_min_unique_documents == 3
    assert configuration.hybrid_local_min_query_term_coverage == 0.8
