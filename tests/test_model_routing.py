from open_deep_research.configuration import Configuration
from open_deep_research.model_routing import (
    ModelTier,
    select_model_route,
    select_paid_fallback_route,
)


def test_routing_is_backward_compatible_when_disabled():
    configuration = Configuration(base_url="https://legacy.example/v1")

    route = select_model_route(
        configuration,
        node="researcher",
        default_model="openai:legacy",
        default_max_tokens=2048,
    )

    assert route.tier == ModelTier.LEGACY
    assert route.model == "openai:legacy"
    assert route.base_url == "https://legacy.example/v1"


def test_low_risk_nodes_use_local_deployment_profile():
    configuration = Configuration(
        enable_adaptive_model_routing=True,
        local_deployment_model="openai:local-model",
        local_deployment_base_url="http://localhost:8000/v1",
        paid_api_model="openai:paid-model",
    )

    route = select_model_route(
        configuration,
        node="researcher",
        default_model="openai:legacy",
        default_max_tokens=4096,
    )

    assert route.tier == ModelTier.LOCAL_DEPLOYMENT
    assert route.model == "openai:local-model"
    assert route.reason == "local_first_node"


def test_high_risk_nodes_use_paid_api_profile():
    configuration = Configuration(
        enable_adaptive_model_routing=True,
        local_deployment_model="openai:local-model",
        paid_api_model="openai:paid-model",
        paid_api_base_url="https://paid.example/v1",
        paid_api_nodes=["final_report_generation"],
    )

    route = select_model_route(
        configuration,
        node="final_report_generation",
        default_model="openai:legacy",
        default_max_tokens=8192,
    )

    assert route.tier == ModelTier.PAID_API
    assert route.model == "openai:paid-model"
    assert route.reason == "high_risk_node"


def test_paid_api_is_fallback_when_local_profile_is_missing():
    configuration = Configuration(
        enable_adaptive_model_routing=True,
        paid_api_model="openai:paid-model",
        paid_api_nodes=[],
    )

    route = select_model_route(
        configuration,
        node="researcher",
        default_model="openai:legacy",
        default_max_tokens=1024,
    )

    assert route.tier == ModelTier.PAID_API
    assert route.reason == "local_profile_unavailable"


def test_failed_local_node_can_escalate_without_rerouting_whole_run():
    configuration = Configuration(
        enable_adaptive_model_routing=True,
        local_deployment_model="openai:local-model",
        paid_api_model="openai:paid-model",
    )
    primary = select_model_route(
        configuration,
        node="researcher",
        default_model="openai:legacy",
        default_max_tokens=2048,
    )

    fallback = select_paid_fallback_route(
        configuration,
        primary=primary,
        default_max_tokens=2048,
    )

    assert fallback is not None
    assert fallback.tier == ModelTier.PAID_API
    assert fallback.reason == "local_node_failure"
