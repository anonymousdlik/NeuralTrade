"""Trading graph — orchestrates all agents in the trading pipeline."""

from __future__ import annotations

import asyncio
import json
from typing import Any

from langgraph.graph import StateGraph, END

from neuraltrade.agents import (
    TradingState, AgentRole,
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

    def _build_graph(self) -> StateGraph:
        """Build the LangGraph trading pipeline."""
        graph = StateGraph(TradingState)

        # Analyst phase (parallel)
        graph.add_node("fundamental_analyst", self._run_fundamental)
        graph.add_node("sentiment_analyst", self._run_sentiment)
        graph.add_node("news_analyst", self._run_news)
        graph.add_node("technical_analyst", self._run_technical)

        # Researcher phase (parallel)
        graph.add_node("bull_researcher", self._run_bull)
        graph.add_node("bear_researcher", self._run_bear)

        # Decision phase (sequential)
        graph.add_node("trader", self._run_trader)
        graph.add_node("risk_manager", self._run_risk)
        graph.add_node("portfolio_manager", self._run_portfolio)

        # Execution
        graph.add_node("execute", self._execute_trade)

        # Edges — analysts run in parallel after start
        graph.set_entry_point("fundamental_analyst")
        graph.add_edge("fundamental_analyst", "sentiment_analyst")
        graph.add_edge("sentiment_analyst", "news_analyst")
        graph.add_edge("news_analyst", "technical_analyst")

        # After all analysts, run researchers
        graph.add_edge("technical_analyst", "bull_researcher")
        graph.add_edge("technical_analyst", "bear_researcher")

        # After researchers, trader decides
        graph.add_edge("bull_researcher", "trader")
        graph.add_edge("bear_researcher", "trader")

        # Risk then portfolio manager
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
        # Try to parse JSON
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
        # Live trading would go here
        return {"order_executed": False, "order_details": {"error": "Live trading not implemented"}}

    async def analyze(self, symbol: str, asset_type: str = "stock", date: str = "") -> TradingState:
        """Run the full trading pipeline for a symbol."""
        initial_state = TradingState(symbol=symbol, asset_type=asset_type, date=date)
        result = await self.graph.ainvoke(initial_state)
        return TradingState(**result)
