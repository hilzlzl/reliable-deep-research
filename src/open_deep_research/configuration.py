"""Configuration management for the Open Deep Research system."""

import os
from enum import Enum
from typing import Any, List, Optional

from langchain_core.runnables import RunnableConfig
from pydantic import BaseModel, Field


class SearchAPI(Enum):
    """Enumeration of available search API providers."""
    
    ANTHROPIC = "anthropic"
    OPENAI = "openai"
    TAVILY = "tavily"
    LOCAL = "local"
    HYBRID = "hybrid"
    NONE = "none"

class MCPConfig(BaseModel):
    """Configuration for Model Context Protocol (MCP) servers."""
    
    url: Optional[str] = Field(
        default=None,
        optional=True,
    )
    """The URL of the MCP server"""
    tools: Optional[List[str]] = Field(
        default=None,
        optional=True,
    )
    """The tools to make available to the LLM"""
    auth_required: Optional[bool] = Field(
        default=False,
        optional=True,
    )
    """Whether the MCP server requires authentication"""

class Configuration(BaseModel):
    """Main configuration class for the Deep Research agent."""
    
    # General Configuration
    max_structured_output_retries: int = Field(
        default=3,
        metadata={
            "x_oap_ui_config": {
                "type": "number",
                "default": 3,
                "min": 1,
                "max": 10,
                "description": "Maximum number of retries for structured output calls from models"
            }
        }
    )
    allow_clarification: bool = Field(
        default=True,
        metadata={
            "x_oap_ui_config": {
                "type": "boolean",
                "default": True,
                "description": "Whether to allow the researcher to ask the user clarifying questions before starting research"
            }
        }
    )
    # Local runtime tracing.  This is independent of LangSmith and is useful
    # for replay, failure injection, and baseline comparisons.
    observability_enabled: bool = Field(default=True)
    event_log_path: str = Field(default="artifacts/research_events.jsonl")
    run_id: Optional[str] = Field(default=None)
    experiment_profile: str = Field(default="default")
    max_llm_calls: Optional[int] = Field(default=None, ge=1)
    max_search_calls: Optional[int] = Field(default=None, ge=1)
    max_task_attempts: Optional[int] = Field(default=None, ge=1)
    max_prompt_tokens: Optional[int] = Field(default=None, ge=1)
    max_completion_tokens: Optional[int] = Field(default=None, ge=1)
    max_total_seconds: Optional[float] = Field(default=None, gt=0)
    base_url: Optional[str] = Field(
        default=None,
        description="Optional OpenAI-compatible API base URL, for example https://gateway.example/v1",
    )
    model_api_key_env: Optional[str] = Field(
        default=None,
        pattern=r"^[A-Z][A-Z0-9_]*$",
        description="Optional environment variable containing the model API key",
    )
    enable_adaptive_model_routing: bool = Field(default=False)
    enable_model_failure_escalation: bool = Field(default=True)
    local_deployment_model: Optional[str] = Field(default=None)
    local_deployment_base_url: Optional[str] = Field(default=None)
    local_deployment_api_key_env: Optional[str] = Field(
        default=None, pattern=r"^[A-Z][A-Z0-9_]*$"
    )
    local_deployment_max_tokens: Optional[int] = Field(default=None, ge=1)
    paid_api_model: Optional[str] = Field(default=None)
    paid_api_base_url: Optional[str] = Field(default=None)
    paid_api_key_env: Optional[str] = Field(
        default=None, pattern=r"^[A-Z][A-Z0-9_]*$"
    )
    paid_api_max_tokens: Optional[int] = Field(default=None, ge=1)
    paid_api_nodes: List[str] = Field(
        default_factory=lambda: [
            "final_report_generation",
            "report_review",
            "report_sentence_audit",
        ]
    )
    runtime_db_path: Optional[str] = Field(
        default=None,
        description="Optional SQLite path for durable task state and results",
    )
    native_structured_output: bool = Field(
        default=True,
        description="Use provider-native structured output before JSON text fallback",
    )
    max_search_queries: int = Field(default=3, ge=1, le=10)
    max_search_results: int = Field(default=5, ge=1, le=20)
    local_corpus_path: Optional[str] = Field(
        default=None,
        description="Directory containing documents.jsonl and chunks.jsonl for local search",
    )
    local_retrieval_strategy: str = Field(
        default="bm25",
        pattern=r"^(bm25|tfidf_vector|rrf_hybrid|rrf_mmr|semantic_vector|bm25_semantic_rrf|semantic_rrf_mmr)$",
    )
    local_embedding_model: str = Field(default="BAAI/bge-small-en-v1.5")
    local_embedding_threads: int = Field(default=8, ge=1)
    local_model_cache_path: str = Field(default="benchmark_data/models")
    local_embedding_cache_path: Optional[str] = Field(
        default=None,
        description="Optional .npz cache; defaults beside the configured local corpus",
    )
    local_retrieval_candidate_k: int = Field(default=40, ge=1)
    local_retrieval_rrf_k: int = Field(default=60, ge=1)
    local_retrieval_mmr_lambda: float = Field(default=0.7, ge=0.0, le=1.0)
    hybrid_local_min_results: int = Field(default=3, ge=1)
    hybrid_local_min_unique_documents: int = Field(default=2, ge=1)
    hybrid_local_min_query_term_coverage: float = Field(
        default=0.75, ge=0.0, le=1.0
    )
    hybrid_min_facet_coverage: float = Field(default=1.0, ge=0.0, le=1.0)
    hybrid_min_candidate_relevance: float = Field(default=0.35, ge=0.0, le=1.0)
    hybrid_min_facet_relevance: float = Field(default=0.55, ge=0.0, le=1.0)
    hybrid_min_source_quality: float = Field(default=0.35, ge=0.0, le=1.0)
    hybrid_max_chunks_per_document: int = Field(default=2, ge=1, le=10)
    hybrid_min_new_documents: int = Field(default=1, ge=0)
    hybrid_min_query_term_coverage_gain: float = Field(
        default=0.05, ge=0.0, le=1.0
    )
    hybrid_max_consecutive_low_gain: int = Field(default=2, ge=1, le=10)
    hybrid_fusion_rrf_k: int = Field(default=60, ge=1)
    fault_injection_rules: List[dict[str, Any]] = Field(default_factory=list)
    replay_task_ids: List[str] = Field(default_factory=list)
    enable_reliable_runtime: bool = Field(
        default=True,
        description="Use the durable DAG scheduler instead of baseline asyncio.gather",
    )
    enable_claim_extraction: bool = Field(
        default=False,
        description="Run one additional claim-extraction model call per research task",
    )
    enable_claim_evidence_validation: bool = Field(
        default=False,
        description="Validate semantic entailment before claims enter a structured report",
    )
    research_task_timeout_seconds: float = Field(default=120.0, gt=0)
    claim_extraction_timeout_seconds: float = Field(default=30.0, gt=0)
    enable_evidence_context: bool = Field(
        default=False,
        description="Generate the final report from budgeted persisted evidence",
    )
    evidence_context_max_characters: int = Field(default=30000, ge=1000)
    enable_report_review: bool = Field(default=False)
    enable_structured_report: bool = Field(
        default=False,
        description="Use a claim-first report plan and deterministic evidence citations",
    )
    enable_gap_search: bool = Field(
        default=False,
        description="Allow one targeted research task for gaps found by the report plan",
    )
    enable_question_coverage: bool = Field(
        default=False,
        description=(
            "Opt into question-first planning, semantic block checks, and one verified "
            "local repair pass (coverage_v1); original profiles retain their behavior"
        ),
    )
    enable_coverage_feedback_repair: bool = Field(
        default=False,
        description="Repair from audit feedback with bounded follow-up search (coverage_v2)",
    )
    coverage_max_repair_rounds: int = Field(default=2, ge=1, le=3)
    coverage_max_gap_searches: int = Field(default=2, ge=0, le=3)
    structured_json_repair_attempts: int = Field(default=0, ge=0, le=2)
    enable_adaptive_research: bool = Field(
        default=False,
        description="Run bounded evidence-aware research rounds before report generation",
    )
    enable_agentic_research_policy: bool = Field(
        default=True,
        description=(
            "Let a structured model policy choose adaptive actions from state and "
            "observed outcomes; invalid decisions fall back to the deterministic policy"
        ),
    )
    max_adaptive_rounds: int = Field(default=2, ge=1, le=5)
    max_adaptive_actions_per_round: int = Field(default=2, ge=1, le=5)
    max_objective_search_attempts: int = Field(default=2, ge=1, le=5)
    adaptive_min_action_utility: float = Field(default=0.15, ge=0.0, le=1.0)
    adaptive_min_evidence_gain: int = Field(default=1, ge=0)
    adaptive_budget_reserve_fraction: float = Field(default=0.2, ge=0.0, lt=1.0)
    enable_sentence_audit: bool = Field(
        default=False,
        description="Audit final factual assertions against their cited evidence",
    )
    sentence_audit_apply_repairs: bool = Field(default=True)
    sentence_audit_min_support_rate: float = Field(
        default=0.6,
        ge=0.0,
        le=1.0,
        description="Minimum independently audited support rate before a report is marked degraded",
    )
    enable_text_tool_call_fallback: bool = Field(
        default=False,
        description="Parse native call:Tool{args} text from compatible local models",
    )
    report_review_max_rounds: int = Field(default=2, ge=0, le=5)
    report_quality_threshold: float = Field(default=0.8, ge=0.0, le=1.0)
    report_min_score_improvement: float = Field(default=0.02, ge=0.0, le=1.0)
    max_concurrent_research_units: int = Field(
        default=5,
        metadata={
            "x_oap_ui_config": {
                "type": "slider",
                "default": 5,
                "min": 1,
                "max": 20,
                "step": 1,
                "description": "Maximum number of research units to run concurrently. This will allow the researcher to use multiple sub-agents to conduct research. Note: with more concurrency, you may run into rate limits."
            }
        }
    )
    max_total_research_units: int = Field(
        default=10,
        ge=1,
        description="Maximum research units scheduled across all supervisor rounds",
    )
    max_initial_research_units: Optional[int] = Field(
        default=None,
        ge=1,
        description="Optional cap before the single coverage-aware gap-search task",
    )
    # Research Configuration
    search_api: SearchAPI = Field(
        default=SearchAPI.TAVILY,
        metadata={
            "x_oap_ui_config": {
                "type": "select",
                "default": "tavily",
                "description": "Search API to use for research. NOTE: Make sure your Researcher Model supports the selected search API.",
                "options": [
                    {"label": "Tavily", "value": SearchAPI.TAVILY.value},
                    {"label": "Local Corpus", "value": SearchAPI.LOCAL.value},
                    {"label": "Adaptive Local + Web", "value": SearchAPI.HYBRID.value},
                    {"label": "OpenAI Native Web Search", "value": SearchAPI.OPENAI.value},
                    {"label": "Anthropic Native Web Search", "value": SearchAPI.ANTHROPIC.value},
                    {"label": "None", "value": SearchAPI.NONE.value}
                ]
            }
        }
    )
    max_researcher_iterations: int = Field(
        default=6,
        metadata={
            "x_oap_ui_config": {
                "type": "slider",
                "default": 6,
                "min": 1,
                "max": 10,
                "step": 1,
                "description": "Maximum number of research iterations for the Research Supervisor. This is the number of times the Research Supervisor will reflect on the research and ask follow-up questions."
            }
        }
    )
    max_react_tool_calls: int = Field(
        default=10,
        metadata={
            "x_oap_ui_config": {
                "type": "slider",
                "default": 10,
                "min": 1,
                "max": 30,
                "step": 1,
                "description": "Maximum number of tool calling iterations to make in a single researcher step."
            }
        }
    )
    # Model Configuration
    summarization_model: str = Field(
        default="openai:gpt-4.1-mini",
        metadata={
            "x_oap_ui_config": {
                "type": "text",
                "default": "openai:gpt-4.1-mini",
                "description": "Model for summarizing research results from Tavily search results"
            }
        }
    )
    summarization_model_max_tokens: int = Field(
        default=8192,
        metadata={
            "x_oap_ui_config": {
                "type": "number",
                "default": 8192,
                "description": "Maximum output tokens for summarization model"
            }
        }
    )
    max_content_length: int = Field(
        default=50000,
        metadata={
            "x_oap_ui_config": {
                "type": "number",
                "default": 50000,
                "min": 1000,
                "max": 200000,
                "description": "Maximum character length for webpage content before summarization"
            }
        }
    )
    research_model: str = Field(
        default="openai:gpt-4.1",
        metadata={
            "x_oap_ui_config": {
                "type": "text",
                "default": "openai:gpt-4.1",
                "description": "Model for conducting research. NOTE: Make sure your Researcher Model supports the selected search API."
            }
        }
    )
    research_model_max_tokens: int = Field(
        default=10000,
        metadata={
            "x_oap_ui_config": {
                "type": "number",
                "default": 10000,
                "description": "Maximum output tokens for research model"
            }
        }
    )
    compression_model: str = Field(
        default="openai:gpt-4.1",
        metadata={
            "x_oap_ui_config": {
                "type": "text",
                "default": "openai:gpt-4.1",
                "description": "Model for compressing research findings from sub-agents. NOTE: Make sure your Compression Model supports the selected search API."
            }
        }
    )
    compression_model_max_tokens: int = Field(
        default=8192,
        metadata={
            "x_oap_ui_config": {
                "type": "number",
                "default": 8192,
                "description": "Maximum output tokens for compression model"
            }
        }
    )
    final_report_model: str = Field(
        default="openai:gpt-4.1",
        metadata={
            "x_oap_ui_config": {
                "type": "text",
                "default": "openai:gpt-4.1",
                "description": "Model for writing the final report from all research findings"
            }
        }
    )
    final_report_model_max_tokens: int = Field(
        default=10000,
        metadata={
            "x_oap_ui_config": {
                "type": "number",
                "default": 10000,
                "description": "Maximum output tokens for final report model"
            }
        }
    )
    # MCP server configuration
    mcp_config: Optional[MCPConfig] = Field(
        default=None,
        optional=True,
        metadata={
            "x_oap_ui_config": {
                "type": "mcp",
                "description": "MCP server configuration"
            }
        }
    )
    mcp_prompt: Optional[str] = Field(
        default=None,
        optional=True,
        metadata={
            "x_oap_ui_config": {
                "type": "text",
                "description": "Any additional instructions to pass along to the Agent regarding the MCP tools that are available to it."
            }
        }
    )


    @classmethod
    def from_runnable_config(
        cls, config: Optional[RunnableConfig] = None
    ) -> "Configuration":
        """Create a Configuration instance from a RunnableConfig."""
        configurable = config.get("configurable", {}) if config else {}
        field_names = list(cls.model_fields.keys())
        values: dict[str, Any] = {
            field_name: configurable.get(field_name, os.environ.get(field_name.upper()))
            for field_name in field_names
        }
        return cls(**{k: v for k, v in values.items() if v is not None})

    class Config:
        """Pydantic configuration."""
        
        arbitrary_types_allowed = True
