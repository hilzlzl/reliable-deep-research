"""Main LangGraph implementation for the Deep Research agent."""

import asyncio
import hashlib
from typing import Any, Literal

from langchain.chat_models import init_chat_model
from langchain_core.messages import (
    AIMessage,
    HumanMessage,
    SystemMessage,
    ToolMessage,
    filter_messages,
    get_buffer_string,
)
from langchain_core.runnables import RunnableConfig
from langgraph.graph import END, START, StateGraph
from langgraph.types import Command

from open_deep_research.configuration import (
    Configuration,
    SearchAPI,
)
from open_deep_research.adaptive_research import (
    ObjectiveStatus,
    PolicyContext,
    ResearchAction,
    ResearchActionType,
    assess_objectives,
    build_research_objectives,
    decide_research_actions,
    propose_research_objectives,
    select_research_actions,
)
from open_deep_research.claim_extraction import (
    extract_and_persist_claims,
    persist_evidence_claims,
)
from open_deep_research.claim_validation import validate_and_persist_claims
from open_deep_research.coverage_reporting import (
    QuestionRequirement,
    decompose_questions,
    generate_coverage_report,
)
from open_deep_research.conflict_detection import detect_and_persist_conflicts
from open_deep_research.context_compression import compress_evidence
from open_deep_research.evidence import audit_report_sources
from open_deep_research.observability import observed_node, record_event
from open_deep_research.runtime_store import RuntimeStore
from open_deep_research.run_budget import (
    BudgetExceeded,
    budget_from_config,
    budgeted_ainvoke,
)
from open_deep_research.report_audit import audit_report_assertions
from open_deep_research.report_review import run_report_review
from open_deep_research.report_planning import (
    create_report_plan,
    deterministic_report_plan,
    deterministic_report_from_plan,
    plan_context,
    render_grounded_report,
)
from open_deep_research.task_runtime import DagTaskScheduler, ResearchTask, TaskStatus
from open_deep_research.tool_call_compat import normalize_text_tool_calls
from open_deep_research.prompts import (
    clarify_with_user_instructions,
    compress_research_simple_human_message,
    compress_research_system_prompt,
    final_report_generation_prompt,
    lead_researcher_prompt,
    research_system_prompt,
    transform_messages_into_research_topic_prompt,
)
from open_deep_research.state import (
    AgentInputState,
    AgentState,
    ClarifyWithUser,
    ConductResearch,
    ResearchComplete,
    ResearcherOutputState,
    ResearcherState,
    ResearchQuestion,
    SupervisorState,
)
from open_deep_research.utils import (
    anthropic_websearch_called,
    get_all_tools,
    get_config_value,
    model_config_for_node,
    paid_fallback_config_for_node,
    get_model_token_limit,
    ainvoke_structured_with_fallback,
    get_notes_from_tool_calls,
    get_today_str,
    is_token_limit_exceeded,
    openai_websearch_called,
    remove_up_to_last_ai_message,
    think_tool,
)

# Initialize a configurable model that we will use throughout the agent
configurable_model = init_chat_model(
    configurable_fields=("model", "max_tokens", "api_key", "base_url"),
)

async def clarify_with_user(state: AgentState, config: RunnableConfig) -> Command[Literal["write_research_brief", "__end__"]]:
    """Analyze user messages and ask clarifying questions if the research scope is unclear.
    
    This function determines whether the user's request needs clarification before proceeding
    with research. If clarification is disabled or not needed, it proceeds directly to research.
    
    Args:
        state: Current agent state containing user messages
        config: Runtime configuration with model settings and preferences
        
    Returns:
        Command to either end with a clarifying question or proceed to research brief
    """
    # Step 1: Check if clarification is enabled in configuration
    configurable = Configuration.from_runnable_config(config)
    if not configurable.allow_clarification:
        # Skip clarification step and proceed directly to research
        return Command(goto="write_research_brief")
    
    # Step 2: Prepare the model for structured clarification analysis
    messages = state["messages"]
    model_config, _ = model_config_for_node(
        configurable,
        config,
        node="clarify_with_user",
        default_model=configurable.research_model,
        default_max_tokens=configurable.research_model_max_tokens,
    )
    
    # Configure model with structured output and retry logic
    clarification_model = (
        configurable_model
        .with_structured_output(ClarifyWithUser)
        .with_retry(stop_after_attempt=configurable.max_structured_output_retries)
        .with_config(model_config)
    )
    
    # Step 3: Analyze whether clarification is needed
    prompt_content = clarify_with_user_instructions.format(
        messages=get_buffer_string(messages), 
        date=get_today_str()
    )
    response = await ainvoke_structured_with_fallback(
        configurable_model.with_config(model_config),
        ClarifyWithUser,
        [HumanMessage(content=prompt_content)],
        retries=configurable.max_structured_output_retries,
        native_structured_output=configurable.native_structured_output,
        config=config,
        node="clarify_with_user",
    )
    
    # Step 4: Route based on clarification analysis
    if response.need_clarification:
        # End with clarifying question for user
        return Command(
            goto=END, 
            update={"messages": [AIMessage(content=response.question)]}
        )
    else:
        # Proceed to research with verification message
        return Command(
            goto="write_research_brief", 
            update={"messages": [AIMessage(content=response.verification)]}
        )


async def write_research_brief(state: AgentState, config: RunnableConfig) -> Command[Literal["research_supervisor"]]:
    """Transform user messages into a structured research brief and initialize supervisor.
    
    This function analyzes the user's messages and generates a focused research brief
    that will guide the research supervisor. It also sets up the initial supervisor
    context with appropriate prompts and instructions.
    
    Args:
        state: Current agent state containing user messages
        config: Runtime configuration with model settings
        
    Returns:
        Command to proceed to research supervisor with initialized context
    """
    # Step 1: Set up the research model for structured output
    configurable = Configuration.from_runnable_config(config)
    research_model_config, _ = model_config_for_node(
        configurable,
        config,
        node="write_research_brief",
        default_model=configurable.research_model,
        default_max_tokens=configurable.research_model_max_tokens,
    )
    
    # Configure model for structured research question generation
    research_model = (
        configurable_model
        .with_structured_output(ResearchQuestion)
        .with_retry(stop_after_attempt=configurable.max_structured_output_retries)
        .with_config(research_model_config)
    )
    
    # Step 2: Generate structured research brief from user messages
    prompt_content = transform_messages_into_research_topic_prompt.format(
        messages=get_buffer_string(state.get("messages", [])),
        date=get_today_str()
    )
    response = await ainvoke_structured_with_fallback(
        configurable_model.with_config(research_model_config),
        ResearchQuestion,
        [HumanMessage(content=prompt_content)],
        retries=configurable.max_structured_output_retries,
        native_structured_output=configurable.native_structured_output,
        config=config,
        node="write_research_brief",
    )

    answer_requirements = []
    if configurable.enable_question_coverage:
        run_budget = budget_from_config(config)
        if run_budget is not None:
            run_budget.set_research_limit_fraction(
                1.0 - configurable.adaptive_budget_reserve_fraction)
            run_budget.set_phase("research")
        original_request = get_buffer_string(
            filter_messages(state.get("messages", []), include_types=["human"])
        )
        try:
            answer_requirements = await decompose_questions(
                configurable_model.with_config(research_model_config), original_request,
                native=configurable.native_structured_output,
                retries=configurable.max_structured_output_retries, config=config,
            )
        except BudgetExceeded:
            raise
        except Exception as error:
            answer_requirements = [QuestionRequirement(
                question_id="q1", question=original_request or response.research_brief)]
            record_event(config, "question_decomposition_fallback", "write_research_brief",
                         data={"error_type": type(error).__name__})
        response.research_brief += "\n\nAnswer requirements:\n" + "\n".join(
            f"- {q.question_id}: {q.question}" for q in answer_requirements)
        record_event(config, "answer_requirements_created", "write_research_brief",
                     data={"questions": [q.model_dump() for q in answer_requirements]})
    
    # Step 3: Initialize supervisor with research brief and instructions
    supervisor_system_prompt = lead_researcher_prompt.format(
        date=get_today_str(),
        max_concurrent_research_units=configurable.max_concurrent_research_units,
        max_researcher_iterations=configurable.max_researcher_iterations
    )
    
    return Command(
        goto="research_supervisor", 
        update={
            "research_brief": response.research_brief,
            "answer_requirements": [q.model_dump() for q in answer_requirements],
            "supervisor_messages": {
                "type": "override",
                "value": [
                    SystemMessage(content=supervisor_system_prompt),
                    HumanMessage(content=response.research_brief)
                ]
            }
        }
    )


async def supervisor(state: SupervisorState, config: RunnableConfig) -> Command[Literal["supervisor_tools"]]:
    """Lead research supervisor that plans research strategy and delegates to researchers.
    
    The supervisor analyzes the research brief and decides how to break down the research
    into manageable tasks. It can use think_tool for strategic planning, ConductResearch
    to delegate tasks to sub-researchers, or ResearchComplete when satisfied with findings.
    
    Args:
        state: Current supervisor state with messages and research context
        config: Runtime configuration with model settings
        
    Returns:
        Command to proceed to supervisor_tools for tool execution
    """
    # Step 1: Configure the supervisor model with available tools
    configurable = Configuration.from_runnable_config(config)
    research_model_config, research_route = model_config_for_node(
        configurable,
        config,
        node="supervisor",
        default_model=configurable.research_model,
        default_max_tokens=configurable.research_model_max_tokens,
    )
    
    # Available tools: research delegation, completion signaling, and strategic thinking
    lead_researcher_tools = [ConductResearch, ResearchComplete, think_tool]
    
    # Configure model with tools, retry logic, and model settings
    research_model = (
        configurable_model
        .bind_tools(lead_researcher_tools)
        .with_retry(stop_after_attempt=configurable.max_structured_output_retries)
        .with_config(research_model_config)
    )
    
    # Step 2: Generate supervisor response based on current context
    supervisor_messages = state.get("supervisor_messages", [])
    try:
        response = await budgeted_ainvoke(
            research_model, supervisor_messages, config, node="supervisor"
        )
    except BudgetExceeded:
        raise
    except Exception as error:
        fallback_config = paid_fallback_config_for_node(
            configurable,
            config,
            node="supervisor",
            primary_route=research_route,
            default_max_tokens=configurable.research_model_max_tokens,
            error=error,
        )
        if fallback_config is None:
            raise
        fallback_model = (
            configurable_model.bind_tools(lead_researcher_tools)
            .with_retry(stop_after_attempt=configurable.max_structured_output_retries)
            .with_config(fallback_config)
        )
        response = await budgeted_ainvoke(
            fallback_model, supervisor_messages, config, node="supervisor_paid_fallback"
        )
    if configurable.enable_text_tool_call_fallback:
        normalized = normalize_text_tool_calls(response, lead_researcher_tools)
        if normalized is not response:
            response = normalized
            record_event(
                config,
                "text_tool_calls_parsed",
                "supervisor",
                data={"tool_names": [item["name"] for item in response.tool_calls]},
            )
    
    # Step 3: Update state and proceed to tool execution
    return Command(
        goto="supervisor_tools",
        update={
            "supervisor_messages": [response],
            "research_iterations": state.get("research_iterations", 0) + 1
        }
    )

async def supervisor_tools(state: SupervisorState, config: RunnableConfig) -> Command[Literal["supervisor", "__end__"]]:
    """Execute tools called by the supervisor, including research delegation and strategic thinking.
    
    This function handles three types of supervisor tool calls:
    1. think_tool - Strategic reflection that continues the conversation
    2. ConductResearch - Delegates research tasks to sub-researchers
    3. ResearchComplete - Signals completion of research phase
    
    Args:
        state: Current supervisor state with messages and iteration count
        config: Runtime configuration with research limits and model settings
        
    Returns:
        Command to either continue supervision loop or end research phase
    """
    # Step 1: Extract current state and check exit conditions
    configurable = Configuration.from_runnable_config(config)
    supervisor_messages = state.get("supervisor_messages", [])
    research_iterations = state.get("research_iterations", 0)
    research_units_started = state.get("research_units_started", 0)
    most_recent_message = supervisor_messages[-1]
    
    # Define exit criteria for research phase
    exceeded_allowed_iterations = research_iterations > configurable.max_researcher_iterations
    no_tool_calls = not most_recent_message.tool_calls
    research_complete_tool_call = any(
        tool_call["name"] == "ResearchComplete" 
        for tool_call in most_recent_message.tool_calls
    )

    # Some OpenAI-compatible models return a plain assistant message instead
    # of selecting ConductResearch. In V2 this must not silently skip all
    # retrieval: when no unit has started yet and local retrieval is available,
    # schedule one conservative task for the original research brief.
    search_api = SearchAPI(get_config_value(configurable.search_api))
    forced_research_call = None
    if (
        (no_tool_calls or research_complete_tool_call or exceeded_allowed_iterations)
        and configurable.enable_reliable_runtime
        and (configurable.experiment_profile in {"reliable_v2", "adaptive_v3"}
             or configurable.enable_question_coverage)
        and search_api in {SearchAPI.LOCAL, SearchAPI.HYBRID}
        and research_units_started == 0
        # A supervisor may spend the first allowed iteration on think_tool;
        # permit the fallback on the immediately following response before
        # applying the normal iteration cutoff.
        and research_iterations <= configurable.max_researcher_iterations + 1
    ):
        forced_research_call = {
            "id": f"fallback-research-{research_iterations}",
            "name": "ConductResearch",
            "args": {"research_topic": state.get("research_brief", "")},
        }
        no_tool_calls = False
        research_complete_tool_call = False
        record_event(
            config,
            "supervisor_research_fallback",
            "supervisor_tools",
            data={"reason": "model_returned_no_tool_call", "research_iterations": research_iterations},
        )
        exceeded_allowed_iterations = False
    
    # Exit if any termination condition is met
    if exceeded_allowed_iterations or no_tool_calls or research_complete_tool_call:
        return Command(
            goto=END,
            update={
                "notes": get_notes_from_tool_calls(supervisor_messages),
                "research_brief": state.get("research_brief", "")
            }
        )
    
    # Step 2: Process all tool calls together (both think_tool and ConductResearch)
    all_tool_messages = []
    update_payload = {"supervisor_messages": []}
    
    # Handle think_tool calls (strategic reflection)
    think_tool_calls = [
        tool_call for tool_call in most_recent_message.tool_calls 
        if tool_call["name"] == "think_tool"
    ]
    
    for tool_call in think_tool_calls:
        reflection_content = tool_call["args"]["reflection"]
        all_tool_messages.append(ToolMessage(
            content=f"Reflection recorded: {reflection_content}",
            name="think_tool",
            tool_call_id=tool_call["id"]
        ))

    if forced_research_call is not None:
        for tool_call in most_recent_message.tool_calls:
            if tool_call["name"] == "ResearchComplete":
                all_tool_messages.append(
                    ToolMessage(
                        content=(
                            "Research completion was deferred because no research unit "
                            "had collected evidence yet."
                        ),
                        name="ResearchComplete",
                        tool_call_id=tool_call["id"],
                    )
                )
    
    # Handle ConductResearch calls (research delegation)
    conduct_research_calls = [
        tool_call for tool_call in most_recent_message.tool_calls 
        if tool_call["name"] == "ConductResearch"
    ]
    if forced_research_call is not None:
        conduct_research_calls.append(forced_research_call)
    
    if conduct_research_calls:
        initial_limit = (
            configurable.max_initial_research_units
            or configurable.max_total_research_units
        )
        initial_limit = min(initial_limit, configurable.max_total_research_units)
        remaining_units = max(0, initial_limit - research_units_started)
        allowed_count = min(
            configurable.max_concurrent_research_units, remaining_units
        )
        allowed_calls = conduct_research_calls[:allowed_count]
        overflow_calls = conduct_research_calls[allowed_count:]
        update_payload["research_units_started"] = (
            research_units_started + len(allowed_calls)
        )
        raw_notes = []
        if configurable.enable_reliable_runtime:
            scheduled_tasks = [
                ResearchTask(
                    task_id=f"research:{tool_call['id']}",
                    description=tool_call["args"]["research_topic"],
                    timeout_seconds=configurable.research_task_timeout_seconds,
                    retry_limit=1,
                )
                for tool_call in allowed_calls
            ]
            calls_by_task_id = {
                f"research:{tool_call['id']}": tool_call for tool_call in allowed_calls
            }

            async def run_research_task(task: ResearchTask):
                tool_call = calls_by_task_id[task.task_id]
                task_config = {
                    **config,
                    "configurable": {
                        **config.get("configurable", {}),
                        "current_task_id": task.task_id,
                    },
                }
                return await researcher_subgraph.ainvoke(
                    {
                        "researcher_messages": [
                            HumanMessage(content=tool_call["args"]["research_topic"])
                        ],
                        "research_topic": tool_call["args"]["research_topic"],
                    },
                    task_config,
                )

            scheduler_result = await DagTaskScheduler(
                max_concurrency=configurable.max_concurrent_research_units
            ).run(scheduled_tasks, run_research_task, config)
            for tool_call in allowed_calls:
                task_id = f"research:{tool_call['id']}"
                task = scheduler_result.tasks[task_id]
                if task.status == TaskStatus.SUCCEEDED:
                    observation = scheduler_result.results[task_id].content
                    content = observation.get(
                        "compressed_research",
                        "Error synthesizing research report: Maximum retries exceeded",
                    )
                    raw_notes.extend(observation.get("raw_notes", []))
                else:
                    content = f"Research task failed: {task.error or task.status.value}"
                if forced_research_call is tool_call:
                    all_tool_messages.append(
                        HumanMessage(
                            content=(
                                "The reliable runtime required one evidence-gathering "
                                f"task before completion. Its result follows:\n\n{content}"
                            )
                        )
                    )
                else:
                    all_tool_messages.append(
                        ToolMessage(content=content, name=tool_call["name"], tool_call_id=tool_call["id"])
                    )
        else:
            # V0 control path: preserve the upstream all-or-nothing gather
            # semantics and do not attach durable task/evidence context.
            try:
                observations = await asyncio.gather(*[
                    researcher_subgraph.ainvoke(
                        {
                            "researcher_messages": [
                                HumanMessage(content=tool_call["args"]["research_topic"])
                            ],
                            "research_topic": tool_call["args"]["research_topic"],
                        },
                        config,
                    )
                    for tool_call in allowed_calls
                ])
            except Exception:
                return Command(
                    goto=END,
                    update={
                        "notes": get_notes_from_tool_calls(supervisor_messages),
                        "research_brief": state.get("research_brief", ""),
                    },
                )
            for observation, tool_call in zip(observations, allowed_calls):
                content = observation.get(
                    "compressed_research",
                    "Error synthesizing research report: Maximum retries exceeded",
                )
                raw_notes.extend(observation.get("raw_notes", []))
                all_tool_messages.append(
                    ToolMessage(content=content, name=tool_call["name"], tool_call_id=tool_call["id"])
                )

        for overflow_call in overflow_calls:
            overflow_content = (
                "Error: research task was not scheduled because the maximum "
                "concurrency or total research-unit budget was reached "
                f"(concurrency={configurable.max_concurrent_research_units}, "
                f"total={configurable.max_total_research_units})."
            )
            if forced_research_call is overflow_call:
                all_tool_messages.append(HumanMessage(content=overflow_content))
            else:
                all_tool_messages.append(
                    ToolMessage(
                        content=overflow_content,
                        name="ConductResearch",
                        tool_call_id=overflow_call["id"],
                    )
                )
        if raw_notes:
            update_payload["raw_notes"] = ["\n".join(raw_notes)]
    
    # Step 3: Return command with all tool results
    update_payload["supervisor_messages"] = all_tool_messages
    return Command(
        goto="supervisor",
        update=update_payload
    ) 

# Supervisor Subgraph Construction
# Creates the supervisor workflow that manages research delegation and coordination
supervisor_builder = StateGraph(SupervisorState, config_schema=Configuration)

# Add supervisor nodes for research management
supervisor_builder.add_node("supervisor", observed_node("supervisor", supervisor))
supervisor_builder.add_node("supervisor_tools", observed_node("supervisor_tools", supervisor_tools))

# Define supervisor workflow edges
supervisor_builder.add_edge(START, "supervisor")  # Entry point to supervisor

# Compile supervisor subgraph for use in main workflow
supervisor_subgraph = supervisor_builder.compile()

async def researcher(state: ResearcherState, config: RunnableConfig) -> Command[Literal["researcher_tools"]]:
    """Individual researcher that conducts focused research on specific topics.
    
    This researcher is given a specific research topic by the supervisor and uses
    available tools (search, think_tool, MCP tools) to gather comprehensive information.
    It can use think_tool for strategic planning between searches.
    
    Args:
        state: Current researcher state with messages and topic context
        config: Runtime configuration with model settings and tool availability
        
    Returns:
        Command to proceed to researcher_tools for tool execution
    """
    # Step 1: Load configuration and validate tool availability
    configurable = Configuration.from_runnable_config(config)
    researcher_messages = state.get("researcher_messages", [])
    
    # Get all available research tools (search, MCP, think_tool)
    tools = await get_all_tools(config)
    if len(tools) == 0:
        raise ValueError(
            "No tools found to conduct research: Please configure either your "
            "search API or add MCP tools to your configuration."
        )
    
    # Step 2: Configure the researcher model with tools
    research_model_config, research_route = model_config_for_node(
        configurable,
        config,
        node="researcher",
        default_model=configurable.research_model,
        default_max_tokens=configurable.research_model_max_tokens,
    )
    
    # Prepare system prompt with MCP context if available
    researcher_prompt = research_system_prompt.format(
        mcp_prompt=configurable.mcp_prompt or "", 
        date=get_today_str()
    )
    
    # Configure model with tools, retry logic, and settings
    research_model = (
        configurable_model
        .bind_tools(tools)
        .with_retry(stop_after_attempt=configurable.max_structured_output_retries)
        .with_config(research_model_config)
    )
    
    # Step 3: Generate researcher response with system context
    messages = [SystemMessage(content=researcher_prompt)] + researcher_messages
    try:
        response = await budgeted_ainvoke(
            research_model, messages, config, node="researcher"
        )
    except BudgetExceeded:
        raise
    except Exception as error:
        fallback_config = paid_fallback_config_for_node(
            configurable,
            config,
            node="researcher",
            primary_route=research_route,
            default_max_tokens=configurable.research_model_max_tokens,
            error=error,
        )
        if fallback_config is None:
            raise
        fallback_model = (
            configurable_model.bind_tools(tools)
            .with_retry(stop_after_attempt=configurable.max_structured_output_retries)
            .with_config(fallback_config)
        )
        response = await budgeted_ainvoke(
            fallback_model, messages, config, node="researcher_paid_fallback"
        )
    if configurable.enable_text_tool_call_fallback:
        normalized = normalize_text_tool_calls(response, tools)
        if normalized is not response:
            response = normalized
            record_event(
                config,
                "text_tool_calls_parsed",
                "researcher",
                data={"tool_names": [item["name"] for item in response.tool_calls]},
            )
    
    # Step 4: Update state and proceed to tool execution
    return Command(
        goto="researcher_tools",
        update={
            "researcher_messages": [response],
            "tool_call_iterations": state.get("tool_call_iterations", 0) + 1
        }
    )

# Tool Execution Helper Function
async def execute_tool_safely(tool, args, config):
    """Safely execute a tool with error handling."""
    try:
        return await tool.ainvoke(args, config)
    except Exception as e:
        return f"Error executing tool: {str(e)}"


async def researcher_tools(state: ResearcherState, config: RunnableConfig) -> Command[Literal["researcher", "compress_research"]]:
    """Execute tools called by the researcher, including search tools and strategic thinking.
    
    This function handles various types of researcher tool calls:
    1. think_tool - Strategic reflection that continues the research conversation
    2. Search tools (tavily_search, web_search) - Information gathering
    3. MCP tools - External tool integrations
    4. ResearchComplete - Signals completion of individual research task
    
    Args:
        state: Current researcher state with messages and iteration count
        config: Runtime configuration with research limits and tool settings
        
    Returns:
        Command to either continue research loop or proceed to compression
    """
    # Step 1: Extract current state and check early exit conditions
    configurable = Configuration.from_runnable_config(config)
    researcher_messages = state.get("researcher_messages", [])
    most_recent_message = researcher_messages[-1]
    
    # Early exit if no tool calls were made (including native web search)
    has_tool_calls = bool(most_recent_message.tool_calls)
    has_native_search = (
        openai_websearch_called(most_recent_message) or 
        anthropic_websearch_called(most_recent_message)
    )

    if not has_tool_calls and not has_native_search:
        search_api = SearchAPI(get_config_value(configurable.search_api))
        if (
            configurable.enable_reliable_runtime
            and search_api in {SearchAPI.LOCAL, SearchAPI.HYBRID}
            and not state.get("search_performed", False)
            and not state.get("local_search_fallback_used", False)
        ):
            tools = await get_all_tools(config)
            local_search_tool = next(
                (
                    item
                    for item in tools
                    if getattr(item, "name", None)
                    in {"local_corpus_search", "adaptive_hybrid_search"}
                ),
                None,
            )
            if local_search_tool is not None:
                topic = state.get("research_topic", "").strip()
                observation = await execute_tool_safely(
                    local_search_tool, {"queries": [topic]}, config
                )
                record_event(
                    config,
                    "local_search_fallback_executed",
                    "researcher_tools",
                    data={"query": topic},
                )
                return Command(
                    goto="compress_research",
                    update={
                        "researcher_messages": [
                            HumanMessage(
                                content=(
                                    "The reliable runtime performed the required local "
                                    "evidence search because no search tool was called. "
                                    "Use only these results for the research summary:\n\n"
                                    f"{observation}"
                                )
                            )
                        ],
                        "local_search_fallback_used": True,
                        "search_performed": True,
                        "fallback_evidence_context": str(observation),
                    },
                )
        return Command(goto="compress_research")
    
    # Step 2: Handle other tool calls (search, MCP tools, etc.)
    tools = await get_all_tools(config)
    tools_by_name = {
        tool.name if hasattr(tool, "name") else tool.get("name", "web_search"): tool 
        for tool in tools
    }
    
    # Execute all tool calls in parallel
    tool_calls = most_recent_message.tool_calls
    tool_execution_tasks = [
        execute_tool_safely(tools_by_name[tool_call["name"]], tool_call["args"], config) 
        for tool_call in tool_calls
    ]
    observations = await asyncio.gather(*tool_execution_tasks)
    
    # Create tool messages from execution results
    tool_outputs = [
        ToolMessage(
            content=observation,
            name=tool_call["name"],
            tool_call_id=tool_call["id"]
        ) 
        for observation, tool_call in zip(observations, tool_calls)
    ]
    
    # Step 3: Check late exit conditions (after processing tools)
    exceeded_iterations = state.get("tool_call_iterations", 0) >= configurable.max_react_tool_calls
    research_complete_called = any(
        tool_call["name"] == "ResearchComplete" 
        for tool_call in most_recent_message.tool_calls
    )
    search_performed = state.get("search_performed", False) or any(
        tool_call["name"] in {"local_corpus_search", "adaptive_hybrid_search"}
        for tool_call in most_recent_message.tool_calls
    )
    
    if exceeded_iterations or research_complete_called:
        # End research and proceed to compression
        return Command(
            goto="compress_research",
            update={
                "researcher_messages": tool_outputs,
                "search_performed": search_performed,
            }
        )
    
    # Continue research loop with tool results
    return Command(
        goto="researcher",
        update={
            "researcher_messages": tool_outputs,
            "search_performed": search_performed,
        }
    )

async def compress_research(state: ResearcherState, config: RunnableConfig):
    """Compress and synthesize research findings into a concise, structured summary.
    
    This function takes all the research findings, tool outputs, and AI messages from
    a researcher's work and distills them into a clean, comprehensive summary while
    preserving all important information and findings.
    
    Args:
        state: Current researcher state with accumulated research messages
        config: Runtime configuration with compression model settings
        
    Returns:
        Dictionary containing compressed research summary and raw notes
    """
    # Step 1: Configure the compression model
    configurable = Configuration.from_runnable_config(config)
    compression_model_config, _ = model_config_for_node(
        configurable,
        config,
        node="compress_research",
        default_model=configurable.compression_model,
        default_max_tokens=configurable.compression_model_max_tokens,
    )
    synthesizer_model = configurable_model.with_config(compression_model_config)
    
    # Step 2: Prepare messages for compression
    researcher_messages = state.get("researcher_messages", [])
    fallback_context = state.get("fallback_evidence_context", "").strip()
    if fallback_context:
        if configurable.enable_claim_extraction and configurable.runtime_db_path:
            task_id = config.get("configurable", {}).get("current_task_id")
            if task_id:
                store = RuntimeStore(configurable.runtime_db_path)
                persist_evidence_claims(store.list_evidence(task_id), store)
                detect_and_persist_conflicts(store)
        record_event(
            config,
            "deterministic_research_fallback",
            "compress_research",
            data={"characters": len(fallback_context)},
        )
        return {
            "compressed_research": fallback_context,
            "raw_notes": [fallback_context],
        }
    
    # Add instruction to switch from research mode to compression mode
    researcher_messages.append(HumanMessage(content=compress_research_simple_human_message))
    
    # Step 3: Attempt compression with retry logic for token limit issues
    synthesis_attempts = 0
    max_attempts = 3
    
    while synthesis_attempts < max_attempts:
        try:
            # Create system prompt focused on compression task
            compression_prompt = compress_research_system_prompt.format(date=get_today_str())
            messages = [SystemMessage(content=compression_prompt)] + researcher_messages
            
            # Execute compression
            response = await budgeted_ainvoke(
                synthesizer_model, messages, config, node="compress_research"
            )

            if configurable.enable_claim_extraction and configurable.runtime_db_path:
                task_id = config.get("configurable", {}).get("current_task_id")
                if task_id:
                    await extract_and_persist_claims(
                        synthesizer_model,
                        task_id=task_id,
                        store=RuntimeStore(configurable.runtime_db_path),
                        native_structured_output=configurable.native_structured_output,
                        retries=configurable.max_structured_output_retries,
                        timeout_seconds=configurable.claim_extraction_timeout_seconds,
                        config=config,
                    )
                    detect_and_persist_conflicts(
                        RuntimeStore(configurable.runtime_db_path)
                    )
            
            # Extract raw notes from all tool and AI messages
            raw_notes_content = "\n".join([
                str(message.content) 
                for message in filter_messages(researcher_messages, include_types=["tool", "ai"])
            ])
            
            # Return successful compression result
            return {
                "compressed_research": str(response.content),
                "raw_notes": [raw_notes_content]
            }
            
        except BudgetExceeded:
            raise
        except Exception as e:
            synthesis_attempts += 1
            
            # Handle token limit exceeded by removing older messages
            if is_token_limit_exceeded(e, configurable.research_model):
                researcher_messages = remove_up_to_last_ai_message(researcher_messages)
                continue
            
            # For other errors, continue retrying
            continue
    
    # Step 4: Return error result if all attempts failed
    raw_notes_content = "\n".join([
        str(message.content) 
        for message in filter_messages(researcher_messages, include_types=["tool", "ai"])
    ])
    
    return {
        "compressed_research": "Error synthesizing research report: Maximum retries exceeded",
        "raw_notes": [raw_notes_content]
    }

# Researcher Subgraph Construction
# Creates individual researcher workflow for conducting focused research on specific topics
researcher_builder = StateGraph(
    ResearcherState, 
    output=ResearcherOutputState, 
    config_schema=Configuration
)

# Add researcher nodes for research execution and compression
researcher_builder.add_node("researcher", observed_node("researcher", researcher))
researcher_builder.add_node("researcher_tools", observed_node("researcher_tools", researcher_tools))
researcher_builder.add_node("compress_research", observed_node("compress_research", compress_research))

# Define researcher workflow edges
researcher_builder.add_edge(START, "researcher")           # Entry point to researcher
researcher_builder.add_edge("compress_research", END)      # Exit point after compression

# Compile researcher subgraph for parallel execution by supervisor
researcher_subgraph = researcher_builder.compile()


async def _run_single_gap_search(
    gaps: list[str], config: RunnableConfig, configurable: Configuration
) -> bool:
    """Run at most one durable, targeted research task for report coverage gaps."""
    if not gaps or not configurable.runtime_db_path:
        return False
    store = RuntimeStore(configurable.runtime_db_path)
    task_count, _ = store.task_counts()
    if task_count >= configurable.max_total_research_units:
        return False
    description = (
        "Resolve only these missing points for the final report. Find direct evidence "
        "and do not broaden the scope:\n- " + "\n- ".join(gaps)
    )
    digest = hashlib.sha256(description.encode("utf-8")).hexdigest()[:16]
    task = ResearchTask(
        task_id=f"gap:{digest}",
        description=description,
        timeout_seconds=configurable.research_task_timeout_seconds,
        retry_limit=1,
    )

    async def run_gap_task(current: ResearchTask):
        task_config = {
            **config,
            "configurable": {
                **config.get("configurable", {}),
                "current_task_id": current.task_id,
            },
        }
        return await researcher_subgraph.ainvoke(
            {
                "researcher_messages": [HumanMessage(content=description)],
                "research_topic": description,
            },
            task_config,
        )

    result = await DagTaskScheduler(max_concurrency=1).run([task], run_gap_task, config)
    succeeded = result.tasks[task.task_id].status == TaskStatus.SUCCEEDED
    record_event(
        config,
        "coverage_gap_search_finished",
        "final_report_generation",
        data={"task_id": task.task_id, "gaps": gaps, "succeeded": succeeded},
    )
    return succeeded


def _adaptive_task_config(
    config: RunnableConfig,
    configurable: Configuration,
    action: ResearchAction,
    task_id: str,
) -> RunnableConfig:
    """Apply a per-action retrieval and model route without mutating run config."""
    values = dict(config.get("configurable", {}))
    values["current_task_id"] = task_id
    values["search_api"] = action.retrieval_route
    paid_nodes = set(configurable.paid_api_nodes)
    routed_nodes = {"researcher", "compress_research", "claim_extraction"}
    if action.model_tier == "paid_api":
        paid_nodes.update(routed_nodes)
    else:
        paid_nodes.difference_update(routed_nodes)
    values["paid_api_nodes"] = sorted(paid_nodes)
    return {**config, "configurable": values}


async def _run_adaptive_research_loop(
    *,
    research_brief: str,
    report_plan: Any,
    claims: list[dict],
    evidence: list[dict],
    store: RuntimeStore,
    writer_model: Any,
    validation_model: Any | None,
    config: RunnableConfig,
    configurable: Configuration,
) -> tuple[list[dict], list[dict], Any]:
    """Close the evidence loop with bounded, persisted incremental research."""
    run_budget = budget_from_config(config)
    if run_budget is not None:
        # Research can consume at most the non-reserved share.  The final
        # report and sentence audit switch back to the full remaining budget.
        run_budget.set_research_limit_fraction(
            1.0 - configurable.adaptive_budget_reserve_fraction
        )
        run_budget.set_phase("research")
    stop_reason = "max_rounds"
    persisted = {
        item["objective_id"]: item for item in store.list_research_objectives()
    }
    if not persisted:
        try:
            proposed = await propose_research_objectives(
                writer_model,
                research_brief=research_brief,
                claims=claims,
                gaps=report_plan.gaps,
                persisted=persisted,
                native_structured_output=configurable.native_structured_output,
                retries=configurable.max_structured_output_retries,
                config=config,
            )
        except Exception as objective_error:
            record_event(
                config,
                "research_objective_planning_fallback",
                "adaptive_research",
                data={"error_type": type(objective_error).__name__},
            )
        else:
            for objective in proposed:
                store.save_research_objective(objective)

    for round_number in range(1, configurable.max_adaptive_rounds + 1):
        persisted = {
            item["objective_id"]: item for item in store.list_research_objectives()
        }
        objectives = build_research_objectives(
            research_brief, claims, report_plan.gaps, persisted
        )
        assessments = assess_objectives(objectives, claims, evidence)
        for assessment in assessments:
            store.save_research_objective(assessment.objective)

        task_count, _ = store.task_counts()
        remaining_slots = max(0, configurable.max_total_research_units - task_count)
        budget = budget_from_config(config)
        policy_context = PolicyContext(
            search_api=SearchAPI(get_config_value(configurable.search_api)),
            local_corpus_available=bool(configurable.local_corpus_path),
            local_model_available=(
                configurable.enable_adaptive_model_routing
                and bool(configurable.local_deployment_model)
            ),
            paid_model_available=(
                configurable.enable_adaptive_model_routing
                and bool(configurable.paid_api_model)
            ),
            remaining_task_slots=remaining_slots,
            max_actions=configurable.max_adaptive_actions_per_round,
            min_utility=configurable.adaptive_min_action_utility,
            # RunBudget enforces this reserve as a hard research-phase ceiling.
            budget_reserve_fraction=0.0,
            max_search_attempts_per_objective=(
                configurable.max_objective_search_attempts
            ),
            budget=budget.snapshot() if budget else None,
        )
        policy_source = "deterministic_fallback"
        policy_candidates: list[dict[str, Any]] = []
        policy_summary = "Deterministic fallback policy selected the next action."
        actions: list[ResearchAction] = []
        if configurable.enable_agentic_research_policy:
            history = []
            for item in store.list_adaptive_decisions()[-12:]:
                payload = item.get("payload", {})
                history.append(
                    {
                        "round": item.get("round_number"),
                        "objective_id": item.get("objective_id"),
                        "action": item.get("action"),
                        "query": payload.get("query"),
                        "expected_quality_gain": payload.get(
                            "expected_quality_gain"
                        ),
                        "expected_cost": payload.get("expected_cost"),
                        "task_status": payload.get("task_status"),
                        "evidence_gain": payload.get("evidence_gain"),
                        "supported_claim_gain": payload.get(
                            "supported_claim_gain"
                        ),
                        "actual_cost": payload.get("actual_cost"),
                    }
                )
            try:
                policy_result = await decide_research_actions(
                    writer_model,
                    research_brief=research_brief,
                    assessments=assessments,
                    context=policy_context,
                    decision_history=history,
                    native_structured_output=configurable.native_structured_output,
                    retries=configurable.max_structured_output_retries,
                    config=config,
                )
            except Exception as policy_error:
                record_event(
                    config,
                    "research_policy_agent_fallback",
                    "adaptive_research",
                    data={
                        "round": round_number,
                        "error_type": type(policy_error).__name__,
                        "error": str(policy_error),
                    },
                )
            else:
                actions = policy_result.actions
                policy_source = policy_result.policy_source
                policy_candidates = [
                    item.model_dump(mode="json")
                    for item in policy_result.candidates
                ]
                policy_summary = policy_result.situation_summary

        if not actions:
            actions = select_research_actions(
                assessments,
                policy_context,
                llm_call_limit=configurable.max_llm_calls,
                search_call_limit=configurable.max_search_calls,
                prompt_token_limit=configurable.max_prompt_tokens,
                completion_token_limit=configurable.max_completion_tokens,
                total_seconds_limit=configurable.max_total_seconds,
            )
            policy_candidates = [item.model_dump(mode="json") for item in actions]
        if not actions:
            stop_reason = "no_positive_utility_action"
            # Keep rejected policy candidates observable.  A stop without this
            # context is indistinguishable from "all objectives are complete"
            # in a post-hoc run report.
            diagnostics: dict[str, Any] = {
                "assessment_summaries": [
                    {
                        "objective_id": item.objective.objective_id,
                        "status": item.objective.status.value,
                        "importance": item.objective.importance,
                        "search_attempts": item.objective.search_attempts,
                        "semantic_support": item.semantic_support,
                        "source_diversity": item.source_diversity,
                        "contradiction_level": item.contradiction_level,
                        "estimated_search_gain": item.estimated_search_gain,
                        "reason": item.reason,
                    }
                    for item in assessments
                ],
                "budget": budget.snapshot().model_dump() if budget else None,
                "remaining_task_slots": remaining_slots,
                "configured_min_utility": configurable.adaptive_min_action_utility,
            }
            # Re-run the same deterministic policy with a zero threshold to
            # expose the best rejected candidates without spending a model call.
            if configurable.adaptive_min_action_utility > 0:
                diagnostic_context = PolicyContext.model_validate(
                    {
                        **policy_context.model_dump(),
                        "min_utility": 0.0,
                    }
                )
                diagnostic_actions = select_research_actions(
                    assessments,
                    diagnostic_context,
                    llm_call_limit=configurable.max_llm_calls,
                    search_call_limit=configurable.max_search_calls,
                    prompt_token_limit=configurable.max_prompt_tokens,
                    completion_token_limit=configurable.max_completion_tokens,
                    total_seconds_limit=configurable.max_total_seconds,
                )
                diagnostics["top_rejected_candidates"] = [
                    {
                        "objective_id": item.objective_id,
                        "action": item.action.value,
                        "utility": item.utility,
                        "reason": item.reason,
                    }
                    for item in diagnostic_actions[:5]
                ]
            record_event(
                config,
                "adaptive_research_stopped",
                "adaptive_research",
                data={"round": round_number, "reason": stop_reason, **diagnostics},
            )
            break

        terminal_actions = [
            item for item in actions
            if item.action in {
                ResearchActionType.STOP,
                ResearchActionType.DROP_OBJECTIVE,
                ResearchActionType.WRITE_WITH_CAVEAT,
            }
        ]
        if terminal_actions:
            for action in terminal_actions:
                if action.action == ResearchActionType.DROP_OBJECTIVE:
                    objective = next(
                        (item for item in objectives if item.objective_id == action.objective_id),
                        None,
                    )
                    if objective is not None:
                        objective.status = ObjectiveStatus.ABANDONED
                        store.save_research_objective(objective)
                store.save_adaptive_decision(
                    decision_id=f"decision:terminal:{round_number}:{action.objective_id}",
                    round_number=round_number,
                    objective_id=action.objective_id,
                    action=action.action.value,
                    payload={
                        **action.model_dump(mode="json"),
                        "terminal": True,
                        "policy_source": policy_source,
                        "situation_summary": policy_summary,
                        "policy_candidates": policy_candidates,
                    },
                )
            stop_reason = (
                "all_objectives_sufficient"
                if all(item.action == ResearchActionType.STOP for item in terminal_actions)
                else "terminal_objective_decision"
            )
            record_event(
                config,
                "adaptive_research_terminal_decision",
                "adaptive_research",
                data={
                    "round": round_number,
                    "stop_reason": stop_reason,
                    "actions": [item.model_dump(mode="json") for item in terminal_actions],
                },
            )
            break

        objectives_by_id = {item.objective_id: item for item in objectives}
        tasks: list[ResearchTask] = []
        actions_by_task: dict[str, ResearchAction] = {}
        action_instructions = {
            ResearchActionType.SEARCH_LOCAL: (
                "Search the local corpus for direct evidence that resolves the objective."
            ),
            ResearchActionType.SEARCH_WEB: (
                "Search beyond the current evidence set for direct primary evidence."
            ),
            ResearchActionType.DIVERSIFY_SOURCES: (
                "Seek independent authors, institutions, datasets, or primary sources "
                "that reduce source monoculture."
            ),
            ResearchActionType.RESOLVE_CONFLICT: (
                "Investigate the conflicting definitions, settings, or findings and "
                "preserve the scope in which each source is valid."
            ),
            ResearchActionType.ESCALATE_MODEL: (
                "Use the stronger reasoning route to resolve this consequential objective."
            ),
        }
        for action in actions:
            digest = hashlib.sha256(
                f"{round_number}:{action.action.value}:{action.query}".encode("utf-8")
            ).hexdigest()[:16]
            task_id = f"adaptive:{digest}"
            description = (
                f"{action_instructions[action.action]} Preserve important definitions "
                "and scope limitations, and do not broaden the topic.\n\n"
                f"Objective:\n{action.query}\n\n"
                f"Success condition:\n{action.success_condition or 'Add direct, non-duplicate evidence.'}"
            )
            tasks.append(
                ResearchTask(
                    task_id=task_id,
                    description=description,
                    timeout_seconds=configurable.research_task_timeout_seconds,
                    retry_limit=1,
                )
            )
            actions_by_task[task_id] = action
            store.save_adaptive_decision(
                decision_id=f"decision:{task_id}",
                round_number=round_number,
                objective_id=action.objective_id,
                action=action.action.value,
                payload={
                    **action.model_dump(mode="json"),
                    "task_id": task_id,
                    "task_status": "SELECTED",
                    "policy_source": policy_source,
                    "situation_summary": policy_summary,
                    "policy_candidates": policy_candidates,
                },
            )

        async def run_adaptive_task(task: ResearchTask):
            action = actions_by_task[task.task_id]
            task_config = _adaptive_task_config(
                config, configurable, action, task.task_id
            )
            return await researcher_subgraph.ainvoke(
                {
                    "researcher_messages": [HumanMessage(content=task.description)],
                    "research_topic": task.description,
                },
                task_config,
            )

        before_ids = {str(item["evidence_id"]) for item in evidence}
        execution_budget = budget_from_config(config)
        budget_before_execution = (
            execution_budget.snapshot() if execution_budget is not None else None
        )
        scheduler_result = await DagTaskScheduler(
            max_concurrency=configurable.max_adaptive_actions_per_round
        ).run(tasks, run_adaptive_task, config)
        for task in tasks:
            action = actions_by_task[task.task_id]
            objective = objectives_by_id[action.objective_id]
            objective.search_attempts += 1
            store.save_research_objective(objective)
        evidence = store.list_evidence()
        claims = store.list_claims()
        if validation_model is not None:
            await validate_and_persist_claims(
                validation_model,
                store=store,
                native_structured_output=configurable.native_structured_output,
                retries=configurable.max_structured_output_retries,
                config=config,
            )
            claims = store.list_claims()
        try:
            report_plan = await create_report_plan(
                writer_model,
                research_brief=research_brief,
                claims=claims,
                evidence=evidence,
                native_structured_output=configurable.native_structured_output,
                retries=configurable.max_structured_output_retries,
                config=config,
            )
        except Exception as planning_error:
            report_plan = deterministic_report_plan(research_brief, claims, evidence)
            record_event(
                config,
                "adaptive_report_planning_fallback",
                "adaptive_research",
                data={"error_type": type(planning_error).__name__},
            )

        current_persisted = {
            item["objective_id"]: item for item in store.list_research_objectives()
        }
        try:
            refreshed_objectives = await propose_research_objectives(
                writer_model,
                research_brief=research_brief,
                claims=claims,
                gaps=report_plan.gaps,
                persisted=current_persisted,
                native_structured_output=configurable.native_structured_output,
                retries=configurable.max_structured_output_retries,
                config=config,
            )
        except Exception as objective_error:
            record_event(
                config,
                "research_objective_refresh_failed",
                "adaptive_research",
                data={"round": round_number, "error_type": type(objective_error).__name__},
            )
        else:
            for objective in refreshed_objectives:
                store.save_research_objective(objective)

        budget_after_execution = (
            execution_budget.snapshot() if execution_budget is not None else None
        )
        actual_cost = None
        if budget_before_execution is not None and budget_after_execution is not None:
            actual_cost = {
                "llm_calls": (
                    budget_after_execution.llm_calls
                    - budget_before_execution.llm_calls
                ),
                "search_calls": (
                    budget_after_execution.search_calls
                    - budget_before_execution.search_calls
                ),
                "task_attempts": (
                    budget_after_execution.task_attempts
                    - budget_before_execution.task_attempts
                ),
                "estimated_prompt_tokens": (
                    budget_after_execution.estimated_prompt_tokens
                    - budget_before_execution.estimated_prompt_tokens
                ),
                "observed_completion_tokens": (
                    budget_after_execution.observed_completion_tokens
                    - budget_before_execution.observed_completion_tokens
                ),
                "elapsed_seconds": round(
                    budget_after_execution.elapsed_seconds
                    - budget_before_execution.elapsed_seconds,
                    3,
                ),
            }

        new_evidence_ids = {
            str(item["evidence_id"]) for item in evidence
        } - before_ids
        evidence_gain = len(new_evidence_ids)
        # Marginal support must be attributable to evidence added in this
        # round. Re-extracting claims from already persisted chunks is not a
        # research gain even when validation status changes.
        supported_claim_gain = sum(
            str(item.get("status")) == "SUPPORTED"
            and bool(set(map(str, item.get("evidence_ids", []))) & new_evidence_ids)
            for item in claims
        )
        for task in tasks:
            action = actions_by_task[task.task_id]
            task_state = scheduler_result.tasks[task.task_id]
            decision_id = f"decision:{task.task_id}"
            task_evidence_ids = {
                str(item["evidence_id"])
                for item in store.list_evidence(task.task_id)
            }
            task_supported_claim_gain = sum(
                str(item.get("status")) == "SUPPORTED"
                and bool(
                    set(map(str, item.get("evidence_ids", [])))
                    & task_evidence_ids
                )
                for item in claims
            )
            payload = {
                **action.model_dump(mode="json"),
                "task_id": task.task_id,
                "task_status": task_state.status.value,
                "policy_source": policy_source,
                "situation_summary": policy_summary,
                "policy_candidates": policy_candidates,
                "evidence_gain": len(task_evidence_ids - before_ids),
                "supported_claim_gain": task_supported_claim_gain,
                "actual_cost": actual_cost,
                "evidence_gain_for_round": evidence_gain,
                "supported_claim_gain_for_round": supported_claim_gain,
            }
            store.save_adaptive_decision(
                decision_id=decision_id,
                round_number=round_number,
                objective_id=action.objective_id,
                action=action.action.value,
                payload=payload,
            )
        record_event(
            config,
            "adaptive_research_round_finished",
            "adaptive_research",
            data={
                "round": round_number,
                "policy_source": policy_source,
                "situation_summary": policy_summary,
                "candidates": policy_candidates,
                "actions": [item.model_dump(mode="json") for item in actions],
                "evidence_gain": evidence_gain,
                "supported_claim_gain": supported_claim_gain,
                "actual_cost": actual_cost,
                "remaining_gaps": report_plan.gaps,
            },
        )
        original_search_api = SearchAPI(get_config_value(configurable.search_api))
        can_broaden_after_local_miss = (
            original_search_api != SearchAPI.LOCAL
            and any(action.action.value == "SEARCH_LOCAL" for action in actions)
        )
        if (
            policy_source != "agent"
            and evidence_gain < configurable.adaptive_min_evidence_gain
            and not can_broaden_after_local_miss
        ):
            stop_reason = "insufficient_evidence_gain"
            break

    persisted = {
        item["objective_id"]: item for item in store.list_research_objectives()
    }
    final_objectives = build_research_objectives(
        research_brief, claims, report_plan.gaps, persisted
    )
    final_assessments = assess_objectives(final_objectives, claims, evidence)
    for assessment in final_assessments:
        store.save_research_objective(assessment.objective)
    unresolved = [
        item.objective.question
        for item in sorted(
            final_assessments,
            key=lambda item: (-item.objective.importance, item.objective.objective_id),
        )
        if item.objective.status.value != "SUFFICIENT"
    ]
    report_plan.gaps = list(dict.fromkeys([*report_plan.gaps, *unresolved]))[:3]

    if run_budget is not None:
        run_budget.set_phase("final")

    record_event(
        config,
        "adaptive_research_finished",
        "adaptive_research",
        data={
            "stop_reason": stop_reason,
            "objectives": len(store.list_research_objectives()),
            "decisions": len(store.list_adaptive_decisions()),
            "assessment_summaries": [
                {
                    "objective_id": item.objective.objective_id,
                    "status": item.objective.status.value,
                    "importance": item.objective.importance,
                    "semantic_support": item.semantic_support,
                    "estimated_search_gain": item.estimated_search_gain,
                    "reason": item.reason,
                }
                for item in final_assessments
            ],
        },
    )
    return evidence, claims, report_plan


async def final_report_generation(state: AgentState, config: RunnableConfig):
    """Generate the final comprehensive research report with retry logic for token limits.
    
    This function takes all collected research findings and synthesizes them into a 
    well-structured, comprehensive final report using the configured report generation model.
    
    Args:
        state: Agent state containing research findings and context
        config: Runtime configuration with model settings and API keys
        
    Returns:
        Dictionary containing the final report and cleared state
    """
    # Step 1: Extract research findings and prepare state cleanup
    notes = state.get("notes", [])
    cleared_state = {"notes": {"type": "override", "value": []}}
    findings = "\n".join(notes)
    
    # Step 2: Configure the final report generation model
    configurable = Configuration.from_runnable_config(config)
    allowed_source_urls: set[str] = set()
    evidence: list[dict] = []
    claims: list[dict] = []
    store: RuntimeStore | None = None
    report_quality_status = "not_audited"
    if configurable.enable_evidence_context and configurable.runtime_db_path:
        store = RuntimeStore(configurable.runtime_db_path)
        evidence = store.list_evidence()
        claims = store.list_claims()
    if configurable.enable_structured_report and not evidence:
        record_event(
            config,
            "report_grounding_failed",
            "final_report_generation",
            data={"reason": "no_persisted_evidence"},
        )
        report_content = (
            "# Insufficient verified evidence\n\n"
            "The research run did not persist any evidence, so Reliable V2 "
            "refused to generate an ungrounded report."
        )
        return {
            "final_report": report_content,
            "messages": [AIMessage(content=report_content)],
            "report_quality_status": "insufficient_evidence",
            **cleared_state,
        }
    writer_model_config, _ = model_config_for_node(
        configurable,
        config,
        node="final_report_generation",
        default_model=configurable.final_report_model,
        default_max_tokens=configurable.final_report_model_max_tokens,
    )
    writer_model = configurable_model.with_config(writer_model_config)

    if configurable.enable_question_coverage and configurable.enable_structured_report:
        run_budget = budget_from_config(config)
        if run_budget is not None:
            run_budget.set_phase("final")

        async def coverage_gap_search(gaps):
            if run_budget is not None:
                snapshot = run_budget.snapshot()
                search_limit = run_budget.limits["search_calls"]
                if search_limit is not None and snapshot.search_calls >= search_limit:
                    record_event(config, "coverage_gap_search_skipped", "final_report_generation",
                                 data={"reason": "search_budget_exhausted"})
                    return False
                run_budget.set_research_limit_fraction(
                    1.0 - configurable.adaptive_budget_reserve_fraction,
                    reserve_search_calls=not configurable.enable_coverage_feedback_repair)
                run_budget.set_phase("research")
            try:
                if run_budget is not None:
                    run_budget.check_time()
                return await _run_single_gap_search(gaps, config, configurable)
            except BudgetExceeded as error:
                record_event(config, "coverage_gap_search_skipped", "final_report_generation",
                             data={"reason": "reserved_finalization_budget",
                                   "resource": error.resource})
                return False
            finally:
                if run_budget is not None:
                    run_budget.set_phase("final")

        questions = [QuestionRequirement.model_validate(q)
                     for q in state.get("answer_requirements", [])]
        if not questions:
            # Direct calls to this node also derive questions from user input,
            # never from retrieved evidence, report drafts, or benchmark labels.
            request = get_buffer_string(filter_messages(
                state.get("messages", []), include_types=["human"]))
            questions = await decompose_questions(
                writer_model, request or state.get("research_brief", ""),
                native=configurable.native_structured_output,
                retries=configurable.max_structured_output_retries, config=config,
            )
        result = await generate_coverage_report(
            writer_model, questions=questions, evidence=evidence,
            config=config, configurable=configurable,
            gap_search=coverage_gap_search,
            reload_evidence=store.list_evidence,
        )
        return {**result, "messages": [AIMessage(content=result["final_report"])],
                **cleared_state}

    report_plan = None
    validation_model = None
    if evidence and configurable.enable_structured_report:
        if configurable.enable_claim_evidence_validation and store is not None:
            validation_model_config, _ = model_config_for_node(
                configurable,
                config,
                node="claim_evidence_validation",
                default_model=configurable.research_model,
                default_max_tokens=configurable.research_model_max_tokens,
            )
            validation_model = configurable_model.with_config(validation_model_config)
            await validate_and_persist_claims(
                validation_model,
                store=store,
                native_structured_output=configurable.native_structured_output,
                retries=configurable.max_structured_output_retries,
                config=config,
            )
            claims = store.list_claims()
        try:
            report_plan = await create_report_plan(
                writer_model,
                research_brief=state.get("research_brief", ""),
                claims=claims,
                evidence=evidence,
                native_structured_output=configurable.native_structured_output,
                retries=configurable.max_structured_output_retries,
                config=config,
            )
        except Exception as planning_error:
            report_plan = deterministic_report_plan(
                state.get("research_brief", ""), claims, evidence
            )
            record_event(
                config,
                "report_planning_fallback",
                "final_report_generation",
                data={"error_type": type(planning_error).__name__},
            )

        if configurable.enable_adaptive_research:
            evidence, claims, report_plan = await _run_adaptive_research_loop(
                research_brief=state.get("research_brief", ""),
                report_plan=report_plan,
                claims=claims,
                evidence=evidence,
                store=store,
                writer_model=writer_model,
                validation_model=validation_model,
                config=config,
                configurable=configurable,
            )
        elif configurable.enable_gap_search and report_plan.gaps:
            if await _run_single_gap_search(report_plan.gaps, config, configurable):
                evidence = store.list_evidence() if store else evidence
                claims = store.list_claims() if store else claims
                if configurable.enable_claim_evidence_validation and store is not None:
                    await validate_and_persist_claims(
                        validation_model,
                        store=store,
                        native_structured_output=configurable.native_structured_output,
                        retries=configurable.max_structured_output_retries,
                        config=config,
                    )
                    claims = store.list_claims()
                try:
                    report_plan = await create_report_plan(
                        writer_model,
                        research_brief=state.get("research_brief", ""),
                        claims=claims,
                        evidence=evidence,
                        native_structured_output=configurable.native_structured_output,
                        retries=configurable.max_structured_output_retries,
                        config=config,
                    )
                except Exception:
                    report_plan = deterministic_report_plan(
                        state.get("research_brief", ""), claims, evidence
                    )
        findings = plan_context(report_plan, claims, evidence)
        allowed_source_urls = {item["source_url"] for item in evidence}
        planned_claim_count = sum(
            len(item.claim_ids) for item in report_plan.sections
        )
        record_event(
            config,
            "report_plan_created",
            "final_report_generation",
            data={
                "sections": len(report_plan.sections),
                "planned_claims": planned_claim_count,
                "claim_evidence_link_rate": 1.0 if planned_claim_count else None,
                "remaining_gaps": report_plan.gaps,
            },
        )
    elif evidence:
        allowed_source_urls = {item["source_url"] for item in evidence}
        required_ids = {
            evidence_id for claim in claims for evidence_id in claim["evidence_ids"]
        }
        compressed = compress_evidence(
            evidence,
            query=state.get("research_brief", ""),
            max_characters=configurable.evidence_context_max_characters,
            required_evidence_ids=required_ids,
        )
        findings = compressed.context
        record_event(
            config,
            "context_compressed",
            "final_report_generation",
            data=compressed.metrics.model_dump(),
        )
    
    # Step 3: Attempt report generation with token limit retry logic
    max_retries = 3
    current_retry = 0
    findings_token_limit = None
    
    while current_retry <= max_retries:
        try:
            # Create comprehensive prompt with all research context
            final_report_prompt = final_report_generation_prompt.format(
                research_brief=state.get("research_brief", ""),
                messages=get_buffer_string(state.get("messages", [])),
                findings=findings,
                date=get_today_str()
            )
            if report_plan is not None:
                final_report_prompt += """

<Structured Evidence Contract>
The Findings contain a validated report outline, claims, and evidence cards.
- Follow the outline and make only the listed claims.
- Cite supporting facts with the exact marker [EVIDENCE:evidence_id].
- Do not write URLs, Markdown links, a Sources section, or evidence IDs in any
  other format. The application will render citations deterministically.
- Omit unsupported content instead of filling gaps from memory.
</Structured Evidence Contract>
"""
            elif allowed_source_urls:
                allowed_source_list = "\n".join(
                    f"- {url}" for url in sorted(allowed_source_urls)
                )
                final_report_prompt += """

<Evidence Grounding Contract>
The Findings block is the complete allowed evidence set for this report.
- Make factual claims only when they are supported by the Findings.
- Cite only exact Source URLs present in the Findings. Never add a URL, paper,
  benchmark result, or bibliographic reference from memory.
- If the Findings do not support part of the research brief, state that the
  available evidence is insufficient instead of filling the gap from memory.
- Every item in the Sources section must use an exact URL from the Findings.
</Evidence Grounding Contract>

<Allowed Source URLs>
The following is an exact whitelist. Copy URLs verbatim; do not rename paths or
construct new URLs. Any URL absent from this list is prohibited.
{allowed_source_list}
</Allowed Source URLs>
""".format(allowed_source_list=allowed_source_list)
            
            # Generate the final report
            final_report = await budgeted_ainvoke(
                writer_model,
                [HumanMessage(content=final_report_prompt)],
                config,
                node="final_report_generation",
            )
            report_content = str(final_report.content)
            audit_removed_evidence_ids: set[str] = set()
            if configurable.enable_report_review:
                try:
                    outcome = await run_report_review(
                        writer_model,
                        report=report_content,
                        evidence_context=findings,
                        max_rounds=configurable.report_review_max_rounds,
                        quality_threshold=configurable.report_quality_threshold,
                        min_improvement=configurable.report_min_score_improvement,
                        native_structured_output=configurable.native_structured_output,
                        retries=configurable.max_structured_output_retries,
                        config=config,
                    )
                except Exception as review_error:
                    record_event(
                        config,
                        "report_review_failed",
                        "final_report_generation",
                        data={
                            "error_type": type(review_error).__name__,
                            "error": str(review_error),
                            "fallback": "original_report",
                        },
                    )
                else:
                    report_content = outcome.report
                    record_event(
                        config,
                        "report_review_finished",
                        "final_report_generation",
                        data={
                            "final_score": outcome.final_score,
                            "stop_reason": outcome.stop_reason,
                            "rounds": [item.model_dump() for item in outcome.rounds],
                        },
                    )

            if configurable.enable_sentence_audit and evidence and store is not None:
                audit_model_config, _ = model_config_for_node(
                    configurable,
                    config,
                    node="report_sentence_audit",
                    default_model=configurable.research_model,
                    default_max_tokens=configurable.research_model_max_tokens,
                )
                audit_model = configurable_model.with_config(audit_model_config)
                store.replace_report_assertions([])
                try:
                    assertion_audit = await audit_report_assertions(
                        audit_model,
                        report=report_content,
                        evidence=evidence,
                        apply_repairs=configurable.sentence_audit_apply_repairs,
                        native_structured_output=configurable.native_structured_output,
                        retries=configurable.max_structured_output_retries,
                        config=config,
                    )
                except Exception as audit_error:
                    record_event(
                        config,
                        "report_sentence_audit_failed",
                        "final_report_generation",
                        data={"error_type": type(audit_error).__name__},
                    )
                else:
                    report_content = assertion_audit.report
                    support_rate = assertion_audit.support_rate
                    if (
                        support_rate is not None
                        and support_rate < configurable.sentence_audit_min_support_rate
                    ):
                        report_quality_status = "degraded_needs_review"
                        record_event(
                            config,
                            "report_quality_degraded",
                            "final_report_generation",
                            data={
                                "reason": "sentence_support_below_threshold",
                                "support_rate": support_rate,
                                "threshold": configurable.sentence_audit_min_support_rate,
                                "not_supported": assertion_audit.not_supported,
                                "contradicted": assertion_audit.contradicted,
                            },
                        )
                    else:
                        report_quality_status = "audited_pass"
                    audit_removed_evidence_ids = {
                        evidence_id
                        for assertion in assertion_audit.assertions
                        if assertion.repair_action.value == "DELETE"
                        for evidence_id in assertion.evidence_ids
                    }
                    store.replace_report_assertions(assertion_audit.assertions)
                    record_event(
                        config,
                        "report_sentence_audit_finished",
                        "final_report_generation",
                        data=assertion_audit.model_dump(
                            mode="json", exclude={"report", "assertions"}
                        ),
                    )

            if allowed_source_urls and report_plan is None:
                source_audit = audit_report_sources(report_content, allowed_source_urls)
                record_event(
                    config,
                    "report_source_audited",
                    "final_report_generation",
                    data=source_audit.model_dump(),
                )

            if report_plan is not None:
                claims_by_id = {item["claim_id"]: item for item in claims}
                required_evidence_ids = {
                    evidence_id
                    for section in report_plan.sections
                    for claim_id in section.claim_ids
                    for evidence_id in claims_by_id[claim_id]["evidence_ids"]
                } - audit_removed_evidence_ids
                report_content, render_audit = render_grounded_report(
                    report_content,
                    evidence,
                    required_evidence_ids=required_evidence_ids,
                )
                if (
                    (
                        required_evidence_ids
                        and render_audit.citation_coverage != 1.0
                    )
                    or render_audit.unknown_evidence_ids
                ):
                    record_event(
                        config,
                        "report_draft_grounding_fallback",
                        "final_report_generation",
                        data=render_audit.model_dump(),
                    )
                    report_content, render_audit = render_grounded_report(
                        deterministic_report_from_plan(report_plan, claims),
                        evidence,
                        required_evidence_ids=required_evidence_ids,
                    )
                record_event(
                    config,
                    "report_citations_rendered",
                    "final_report_generation",
                    data=render_audit.model_dump(),
                )
                source_audit = audit_report_sources(
                    report_content, allowed_source_urls
                )
                record_event(
                    config,
                    "report_source_audited",
                    "final_report_generation",
                    data=source_audit.model_dump(),
                )

            # Return successful report generation
            return {
                "final_report": report_content,
                "messages": [AIMessage(content=report_content)],
                "report_quality_status": report_quality_status,
                **cleared_state
            }
            
        except BudgetExceeded as error:
            record_event(
                config,
                "run_budget_exhausted",
                "final_report_generation",
                data={"resource": error.resource, "limit": error.limit},
            )
            return {
                "final_report": findings or f"Research stopped: {error}",
                "messages": [AIMessage(content="Research stopped after reaching its configured budget")],
                "report_quality_status": "budget_exhausted",
                **cleared_state,
            }
        except Exception as e:
            # Handle token limit exceeded errors with progressive truncation
            if is_token_limit_exceeded(e, configurable.final_report_model):
                current_retry += 1
                
                if current_retry == 1:
                    # First retry: determine initial truncation limit
                    model_token_limit = get_model_token_limit(configurable.final_report_model)
                    if not model_token_limit:
                        return {
                            "final_report": f"Error generating final report: Token limit exceeded, however, we could not determine the model's maximum context length. Please update the model map in deep_researcher/utils.py with this information. {e}",
                            "messages": [AIMessage(content="Report generation failed due to token limits")],
                            "report_quality_status": "generation_failed",
                            **cleared_state
                        }
                    # Use 4x token limit as character approximation for truncation
                    findings_token_limit = model_token_limit * 4
                else:
                    # Subsequent retries: reduce by 10% each time
                    findings_token_limit = int(findings_token_limit * 0.9)
                
                # Truncate findings and retry
                findings = findings[:findings_token_limit]
                continue
            else:
                # Non-token-limit error: return error immediately
                return {
                    "final_report": f"Error generating final report: {e}",
                    "messages": [AIMessage(content="Report generation failed due to an error")],
                    "report_quality_status": "generation_failed",
                    **cleared_state
                }
    
    # Step 4: Return failure result if all retries exhausted
    return {
        "final_report": "Error generating final report: Maximum retries exceeded",
        "messages": [AIMessage(content="Report generation failed after maximum retries")],
        "report_quality_status": "generation_failed",
        **cleared_state
    }

# Main Deep Researcher Graph Construction
# Creates the complete deep research workflow from user input to final report
deep_researcher_builder = StateGraph(
    AgentState, 
    input=AgentInputState, 
    config_schema=Configuration
)

# Add main workflow nodes for the complete research process
deep_researcher_builder.add_node("clarify_with_user", observed_node("clarify_with_user", clarify_with_user))
deep_researcher_builder.add_node("write_research_brief", observed_node("write_research_brief", write_research_brief))
deep_researcher_builder.add_node("research_supervisor", supervisor_subgraph)       # Research execution phase
deep_researcher_builder.add_node("final_report_generation", observed_node("final_report_generation", final_report_generation))

# Define main workflow edges for sequential execution
deep_researcher_builder.add_edge(START, "clarify_with_user")                       # Entry point
deep_researcher_builder.add_edge("research_supervisor", "final_report_generation") # Research to report
deep_researcher_builder.add_edge("final_report_generation", END)                   # Final exit point

# Compile the complete deep researcher workflow
deep_researcher = deep_researcher_builder.compile()
