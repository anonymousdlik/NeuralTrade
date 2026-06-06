"""Trading graph — orchestrates all agents in the trading pipeline."""

from __future__ import annotations

import asyncio
import json
from typing import Any

from langgraph.graph import StateGraph, END

from neuraltrade.agents import (
    TradingState,
    FUNDAMENTAL_ANALYST_PROMPT, SENTIMENT_ANALYST_PROMPT,
    NEWS_ANALYST_PROMPT, TECHNICAL_ANALYST_PROMPT,
    BULL_RESEARCHER_PROMPT, BEAR_RESEARCHER_PROMPT,
    TRADER_PROMPT, RISK_MANAGER_PROMPT, PORTFOLIO_MANAGER_PROMPT,
)
from neuraltrade.llm import LLMRouter, TaskType


class TradingGraph:
    """Orchestrates the multi-agent trading pipeline."""

    def __init__(self, router: LLMRouter, config: Any = None):
        self.router = router
        self.config = config
        self.graph = self._build_graph()

    def _build_graph(self):
        """Build the LangGraph trading pipeline."""
        graph = StateGraph(TradingState)

        # Analyst phase
        graph.add_node("fundamental_analyst", self._run_fundamental)
        graph.add_node("sentiment_analyst", self._run_sentiment)
        graph.add_node("news_analyst", self._run_news)
        graph.add_node("technical_analyst", self._run_technical)

        # Merge analyst outputs
        graph.add_node("merge_analysts", self._merge_analysts)

        # Researcher phase
        graph.add_node("bull_researcher", self._run_bull)
        graph.add_node("bear_researcher", self._run_bear)

        # Merge researcher outputs
        graph.add_node("merge_researchers", self._merge_researchers)

        # Decision phase (sequential)
        graph.add_node("trader", self._run_trader)
        graph.add_node("risk_manager", self._run_risk)
        graph.add_node("portfolio_manager", self._run_portfolio)

        # Execution
        graph.add_node("execute", self._execute_trade)

        # Entry: all analysts start from START
        graph.set_entry_point("fundamental_analyst")
        graph.set_entry_point("sentiment_analyst")
        graph.set_entry_point("news_analyst")
        graph.set_entry_point("technical_analyst")

        # All analysts -> merge
        graph.add_edge("fundamental_analyst", "merge_analysts")
        graph.add_edge("sentiment_analyst", "merge_analysts")
        graph.add_edge("news_analyst", "merge_analysts")
        graph.add_edge("technical_analyst", "merge_analysts")

        # Merge -> researchers (parallel)
        graph.add_edge("merge_analysts", "bull_researcher")
        graph.add_edge("merge_analysts", "bear_researcher")

        # Researchers -> merge
        graph.add_edge("bull_researcher", "merge_researchers")
        graph.add_edge("bear_researcher", "merge_researchers")

        # Sequential decision chain
        graph.add_edge("merge_researchers", "trader")
        graph.add_edge("trader", "risk_manager")
        graph.add_edge("risk_manager", "portfolio_manager")

        # Conditional execution
        graph.add_conditional_edges(
            "portfolio_manager",
            self._should_execute,
            {"execute": "execute", "skip": END},
        )
        graph.add_edge("execute", END)

        return graph.compile()

    async def _run_fundamental(self, state: TradingState) -> dict:
        prompt = FUNDAMENTAL_ANALYST_PROMPT.format(symbol=state.symbol)
        result = await self.router.chat(
            [{"role": "system", "content": prompt}],
            task=TaskType.STANDARD,
        )
        return {"fundamental_analysis": result}

    async def _run_sentiment(self, state: TradingState) -> dict:
        prompt = SENTIMENT_ANALYST_PROMPT.format(symbol=state.symbol)
        result = await self.router.chat(
            [{"role": "system", "content": prompt}],
            task=TaskType.STANDARD,
        )
        return {"sentiment_analysis": result}

    async def _run_news(self, state: TradingState) -> dict:
        prompt = NEWS_ANALYST_PROMPT.format(symbol=state.symbol)
        result = await self.router.chat(
            [{"role": "system", "content": prompt}],
            task=TaskType.STANDARD,
        )
        return {"news_analysis": result}

    async def _run_technical(self, state: TradingState) -> dict:
        prompt = TECHNICAL_ANALYST_PROMPT.format(symbol=state.symbol)
        result = await self.router.chat(
            [{"role": "system", "content": prompt}],
            task=TaskType.STANDARD,
        )
        return {"technical_analysis": result}

    async def _merge_analysts(self, state: TradingState) -> dict:
        """No-op merge node — waits for all analysts to complete."""
        return {}

    async def _run_bull(self, state: TradingState) -> dict:
        prompt = BULL_RESEARCHER_PROMPT.format(
            symbol=state.symbol,
            fundamental=state.fundamental_analysis,
            sentiment=state.sentiment_analysis,
            news=state.news_analysis,
            technical=state.technical_analysis,
        )
        result = await self.router.chat(
            [{"role": "system", "content": prompt}],
            task=TaskType.COMPLEX,
        )
        return {"bull_case": result}

    async def _run_bear(self, state: TradingState) -> dict:
        prompt = BEAR_RESEARCHER_PROMPT.format(
            symbol=state.symbol,
            fundamental=state.fundamental_analysis,
            sentiment=state.sentiment_analysis,
            news=state.news_analysis,
            technical=state.technical_analysis,
        )
        result = await self.router.chat(
            [{"role": "system", "content": prompt}],
            task=TaskType.COMPLEX,
        )
        return {"bear_case": result}

    async def _merge_researchers(self, state: TradingState) -> dict:
        """No-op merge node — waits for both researchers to complete."""
        return {}

    async def _run_trader(self, state: TradingState) -> dict:
        prompt = TRADER_PROMPT.format(
            symbol=state.symbol,
            bull_case=state.bull_case,
            bear_case=state.bear_case,
        )
        result = await self.router.chat(
            [{"role": "system", "content": prompt}],
            task=TaskType.COMPLEX,
        )
        try:
            data = json.loads(result)
            return {
                "trade_decision": data.get("decision", "HOLD"),
                "trade_amount": data.get("position_size_pct", 0) / 100,
                "trade_reasoning": data.get("reasoning", result),
            }
        except json.JSONDecodeError:
            return {"trade_decision": "HOLD", "trade_amount": 0, "trade_reasoning": result}

    async def _run_risk(self, state: TradingState) -> dict:
        prompt = RISK_MANAGER_PROMPT.format(
            symbol=state.symbol,
            trade_decision=state.trade_reasoning,
            portfolio_state=json.dumps(state.history[-5:] if state.history else []),
        )
        result = await self.router.chat(
            [{"role": "system", "content": prompt}],
            task=TaskType.COMPLEX,
        )
        try:
            data = json.loads(result)
            return {
                "risk_assessment": data.get("reasoning", result),
                "risk_score": data.get("risk_score", 50) / 100,
            }
        except json.JSONDecodeError:
            return {"risk_assessment": result, "risk_score": 0.5}

    async def _run_portfolio(self, state: TradingState) -> dict:
        prompt = PORTFOLIO_MANAGER_PROMPT.format(
            symbol=state.symbol,
            trade_decision=state.trade_reasoning,
            risk_assessment=state.risk_assessment,
        )
        result = await self.router.chat(
            [{"role": "system", "content": prompt}],
            task=TaskType.COMPLEX,
        )
        try:
            data = json.loads(result)
            return {
                "final_decision": data.get("final_decision", "REJECT"),
                "final_reasoning": data.get("reasoning", result),
            }
        except json.JSONDecodeError:
            return {"final_decision": "REJECT", "final_reasoning": result}

    def _should_execute(self, state: TradingState) -> str:
        if state.final_decision == "APPROVE":
            return "execute"
        return "skip"

    async def _execute_trade(self, state: TradingState) -> dict:
        """Execute the trade (paper or live)."""
        if self.config and self.config.trading.paper_trading:
            return {
                "order_executed": True,
                "order_details": {
                    "type": "PAPER",
                    "symbol": state.symbol,
                    "decision": state.trade_decision,
                    "amount": state.trade_amount,
                },
            }
        return {"order_executed": False, "order_details": {"error": "Live trading not implemented"}}

    async def analyze(self, symbol: str, asset_type: str = "stock", date: str = "") -> TradingState:
        """Run the full trading pipeline for a symbol."""
        initial_state = TradingState(symbol=symbol, asset_type=asset_type, date=date)
        result = await self.graph.ainvoke(initial_state)
        return TradingState(**result)
