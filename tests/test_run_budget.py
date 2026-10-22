import asyncio

import pytest
from langchain_core.messages import AIMessage, HumanMessage

from open_deep_research.run_budget import BudgetExceeded, RunBudget, budgeted_ainvoke


class FakeModel:
    def __init__(self):
        self.calls = 0

    async def ainvoke(self, messages):
        self.calls += 1
        return AIMessage(
            content="ok",
            usage_metadata={"input_tokens": 1, "output_tokens": 3, "total_tokens": 4},
        )


def test_budget_rejects_work_before_exceeding_call_limit():
    budget = RunBudget(max_llm_calls=1)
    config = {"configurable": {"run_budget": budget, "observability_enabled": False}}
    model = FakeModel()

    asyncio.run(budgeted_ainvoke(model, [HumanMessage(content="first")], config, node="test"))
    with pytest.raises(BudgetExceeded, match="llm_calls"):
        asyncio.run(budgeted_ainvoke(model, [HumanMessage(content="second")], config, node="test"))

    assert model.calls == 1
    assert budget.snapshot().llm_calls == 1


def test_budget_records_prompt_and_completion_usage():
    budget = RunBudget(max_prompt_tokens=100, max_completion_tokens=100)
    config = {"configurable": {"run_budget": budget, "observability_enabled": False}}

    asyncio.run(
        budgeted_ainvoke(FakeModel(), [HumanMessage(content="a" * 20)], config, node="test")
    )

    snapshot = budget.snapshot()
    assert snapshot.estimated_prompt_tokens == 5
    assert snapshot.observed_completion_tokens == 3


def test_elapsed_budget_times_out_model_call():
    class SlowModel:
        async def ainvoke(self, messages):
            await asyncio.sleep(0.05)

    budget = RunBudget(max_total_seconds=0.001)
    config = {"configurable": {"run_budget": budget, "observability_enabled": False}}

    with pytest.raises(BudgetExceeded, match="total_seconds"):
        asyncio.run(
            budgeted_ainvoke(SlowModel(), [HumanMessage(content="slow")], config, node="test")
        )


def test_llm_reservation_is_atomic_when_prompt_budget_is_exhausted():
    budget = RunBudget(max_llm_calls=2, max_prompt_tokens=1)
    config = {"configurable": {"run_budget": budget, "observability_enabled": False}}
    model = FakeModel()

    with pytest.raises(BudgetExceeded, match="estimated_prompt_tokens"):
        asyncio.run(
            budgeted_ainvoke(model, [HumanMessage(content="too long")], config, node="test")
        )

    assert model.calls == 0
    assert budget.snapshot().llm_calls == 0


def test_research_phase_hard_reserve_leaves_final_budget_available():
    budget = RunBudget(max_llm_calls=10, research_limit_fraction=0.8)
    budget.set_phase("research")
    for _ in range(8):
        budget.consume("llm_calls")
    with pytest.raises(BudgetExceeded, match="llm_calls"):
        budget.consume("llm_calls")
    budget.set_phase("final")
    budget.consume("llm_calls")
    assert budget.snapshot().llm_calls == 9


def test_gap_search_can_use_fourth_search_while_retaining_llm_reserve():
    budget = RunBudget(max_search_calls=4, max_llm_calls=10)
    budget.set_research_limit_fraction(.8)
    budget.set_phase('research')
    budget.consume('search_calls', 3)
    with pytest.raises(BudgetExceeded):
        budget.consume('search_calls')
    budget.set_research_limit_fraction(.8, reserve_search_calls=False)
    budget.consume('search_calls')
    assert budget.snapshot().search_calls == 4
    with pytest.raises(BudgetExceeded):
        budget.consume('search_calls')
    budget.consume('llm_calls', 8)
    with pytest.raises(BudgetExceeded):
        budget.consume('llm_calls')
    budget.set_phase('final')
    budget.consume('llm_calls')
    assert budget.snapshot().llm_calls == 9
