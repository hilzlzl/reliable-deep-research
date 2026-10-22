"""Provider-neutral node-level model routing policy."""

from __future__ import annotations

from enum import Enum
from typing import Any

from pydantic import BaseModel


class ModelTier(str, Enum):
    LOCAL_DEPLOYMENT = "local_deployment"
    PAID_API = "paid_api"
    LEGACY = "legacy"


class ModelRoute(BaseModel):
    tier: ModelTier
    model: str
    base_url: str | None = None
    api_key_env: str | None = None
    max_tokens: int
    reason: str


def select_model_route(
    configurable: Any,
    *,
    node: str,
    default_model: str,
    default_max_tokens: int,
) -> ModelRoute:
    """Select a model profile from node risk without naming any provider."""
    if not configurable.enable_adaptive_model_routing:
        return ModelRoute(
            tier=ModelTier.LEGACY,
            model=default_model,
            base_url=configurable.base_url,
            api_key_env=configurable.model_api_key_env,
            max_tokens=default_max_tokens,
            reason="adaptive_model_routing_disabled",
        )

    paid_nodes = set(configurable.paid_api_nodes)
    if node in paid_nodes and configurable.paid_api_model:
        return ModelRoute(
            tier=ModelTier.PAID_API,
            model=configurable.paid_api_model,
            base_url=configurable.paid_api_base_url,
            api_key_env=configurable.paid_api_key_env,
            max_tokens=(
                configurable.paid_api_max_tokens or default_max_tokens
            ),
            reason="high_risk_node",
        )
    if configurable.local_deployment_model:
        return ModelRoute(
            tier=ModelTier.LOCAL_DEPLOYMENT,
            model=configurable.local_deployment_model,
            base_url=configurable.local_deployment_base_url,
            api_key_env=configurable.local_deployment_api_key_env,
            max_tokens=(
                configurable.local_deployment_max_tokens or default_max_tokens
            ),
            reason="local_first_node",
        )
    if configurable.paid_api_model:
        return ModelRoute(
            tier=ModelTier.PAID_API,
            model=configurable.paid_api_model,
            base_url=configurable.paid_api_base_url,
            api_key_env=configurable.paid_api_key_env,
            max_tokens=configurable.paid_api_max_tokens or default_max_tokens,
            reason="local_profile_unavailable",
        )
    return ModelRoute(
        tier=ModelTier.LEGACY,
        model=default_model,
        base_url=configurable.base_url,
        api_key_env=configurable.model_api_key_env,
        max_tokens=default_max_tokens,
        reason="no_model_profiles_configured",
    )


def select_paid_fallback_route(
    configurable: Any,
    *,
    primary: ModelRoute,
    default_max_tokens: int,
) -> ModelRoute | None:
    """Return a paid fallback only for a failed local-deployment call."""
    if (
        not configurable.enable_adaptive_model_routing
        or not configurable.enable_model_failure_escalation
        or primary.tier != ModelTier.LOCAL_DEPLOYMENT
        or not configurable.paid_api_model
    ):
        return None
    return ModelRoute(
        tier=ModelTier.PAID_API,
        model=configurable.paid_api_model,
        base_url=configurable.paid_api_base_url,
        api_key_env=configurable.paid_api_key_env,
        max_tokens=configurable.paid_api_max_tokens or default_max_tokens,
        reason="local_node_failure",
    )
