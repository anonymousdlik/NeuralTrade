"""CLI interface for NeuralTrade."""

from __future__ import annotations

import asyncio
import json
from typing import Optional

import typer
from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich.progress import Progress, SpinnerColumn, TextColumn

from neuraltrade.config import AppConfig
from neuraltrade.llm import LLMRouter
from neuraltrade.agents.graph import TradingGraph
from neuraltrade.dataflows import MarketDataFlow, NewsDataFlow
from neuraltrade.engine import BacktestEngine

app = typer.Typer(
    name="neuraltrade",
    help="NeuralTrade: Multi-Model AI Trading Framework",
    no_args_is_help=True,
)
console = Console()


def get_config() -> AppConfig:
    return AppConfig.from_env()


def get_router(config: AppConfig) -> LLMRouter:
    return LLMRouter(config)


@app.command()
def analyze(
    symbol: str = typer.Argument(..., help="Stock/crypto symbol (e.g., AAPL, BTC-USD)"),
    asset_type: str = typer.Option("stock", "--type", "-t", help="Asset type: stock or crypto"),
    model: Optional[str] = typer.Option(None, "--model", "-m", help="Override model (provider/model)"),
    verbose: bool = typer.Option(False, "--verbose", "-v", help="Show detailed agent outputs"),
):
    """Run full AI analysis on a symbol."""
    config = get_config()
    if model:
        config.model.default_model = model

    router = get_router(config)
    graph = TradingGraph(router, config)
    data_flow = MarketDataFlow()
    news_flow = NewsDataFlow()

    async def run():
        with Progress(SpinnerColumn(), TextColumn("[progress.description]{task.description}")) as progress:
            # Fetch data
            task = progress.add_task("Fetching market data...", total=None)
            market_data = await data_flow.get_market_data(symbol)
            news_data = await news_flow.get_news(symbol)
            progress.update(task, completed=True)

            # Run agents
            task = progress.add_task("Running AI agents...", total=None)
            result = await graph.analyze(symbol, asset_type)
            progress.update(task, completed=True)

        # Display results
        console.print()
        console.print(Panel(
            f"[bold cyan]{symbol}[/bold cyan] — {asset_type.upper()}\n"
            f"Price: [bold]${market_data.price:,.2f}[/bold]  "
            f"Change: [bold {'green' if market_data.change_pct >= 0 else 'red'}]{market_data.change_pct:+.2f}%[/{'green' if market_data.change_pct >= 0 else 'red'}]\n"
            f"Volume: {market_data.volume:,.0f}  "
            f"Market Cap: ${market_data.market_cap:,.0f}",
            title="📊 Market Data",
            border_style="cyan",
        ))

        if verbose:
            console.print(Panel(result.fundamental_analysis, title="📈 Fundamental Analyst", border_style="blue"))
            console.print(Panel(result.sentiment_analysis, title="💭 Sentiment Analyst", border_style="magenta"))
            console.print(Panel(result.news_analysis, title="📰 News Analyst", border_style="yellow"))
            console.print(Panel(result.technical_analysis, title="📉 Technical Analyst", border_style="green"))
            console.print(Panel(result.bull_case, title="🐂 Bull Researcher", border_style="green"))
            console.print(Panel(result.bear_case, title="🐻 Bear Researcher", border_style="red"))

        # Decision
        decision_color = {"BUY": "green", "SELL": "red", "HOLD": "yellow"}.get(result.trade_decision, "white")
        console.print(Panel(
            f"[bold]Decision: [{decision_color}]{result.trade_decision}[/{decision_color}][/bold]\n\n"
            f"{result.trade_reasoning}\n\n"
            f"Risk Score: [bold]{result.risk_score:.0%}[/bold]\n"
            f"Final: [bold]{result.final_decision}[/bold] — {result.final_reasoning}",
            title="🎯 Trading Decision",
            border_style=decision_color,
        ))

    asyncio.run(run())


@app.command()
def backtest(
    symbol: str = typer.Argument(..., help="Symbol to backtest"),
    period: str = typer.Option("1y", "--period", "-p", help="Backtest period (1mo, 3mo, 6mo, 1y, 2y)"),
    capital: float = typer.Option(100000, "--capital", "-c", help="Initial capital"),
):
    """Backtest a trading strategy on historical data."""
    config = get_config()
    data_flow = MarketDataFlow()
    engine = BacktestEngine(initial_capital=capital)

    async def run():
        market_data = await data_flow.get_market_data(symbol, period=period)
        if market_data.history.empty:
            console.print(f"[red]No data found for {symbol}[/red]")
            return

        # Simple strategy: SMA crossover
        hist = market_data.history
        sma_short = hist["Close"].rolling(20).mean()
        sma_long = hist["Close"].rolling(50).mean()
        signals = pd.Series(0, index=hist.index)
        signals[sma_short > sma_long] = 1
        signals[sma_short <= sma_long] = -1

        result = engine.run(symbol, hist, signals)
        console.print(result.summary())

    import pandas as pd
    asyncio.run(run())


@app.command()
def models():
    """List available LLM models."""
    config = get_config()
    from neuraltrade.llm import list_available_models

    has_keys = {
        "openai": bool(config.openai_key),
        "anthropic": bool(config.anthropic_key),
        "google": bool(config.google_key),
        "openrouter": bool(config.openrouter_key),
        "groq": bool(config.groq_key),
    }

    table = Table(title="🧠 Available Models")
    table.add_column("Model", style="cyan")
    table.add_column("Provider", style="magenta")
    table.add_column("Speed", style="green")
    table.add_column("Quality", style="yellow")
    table.add_column("Key Available", style="red")

    for m in list_available_models():
        key_ok = "✅" if has_keys.get(m.provider.value, False) else "❌"
        stars = "⭐" * m.quality
        table.add_row(
            m.display_name, m.provider.value, m.speed, stars, key_ok
        )

    console.print(table)


@app.command()
def serve(
    host: str = typer.Option("0.0.0.0", "--host", "-h"),
    port: int = typer.Option(8080, "--port", "-p"),
):
    """Start the web dashboard."""
    import uvicorn
    from neuraltrade.dashboard.app import create_app

    console.print(f"[green]Starting NeuralTrade dashboard at http://{host}:{port}[/green]")
    app = create_app()
    uvicorn.run(app, host=host, port=port)


if __name__ == "__main__":
    app()
