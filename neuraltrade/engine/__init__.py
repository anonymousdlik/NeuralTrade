"""Backtesting engine — test strategies on historical data."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any

import pandas as pd
import numpy as np


@dataclass
class Trade:
    """A single trade record."""
    date: str
    symbol: str
    action: str  # BUY or SELL
    price: float
    quantity: float
    value: float
    reason: str = ""


@dataclass
class BacktestResult:
    """Results of a backtest run."""
    symbol: str
    start_date: str
    end_date: str
    initial_capital: float
    final_value: float
    total_return_pct: float
    annualized_return_pct: float
    sharpe_ratio: float
    max_drawdown_pct: float
    win_rate: float
    total_trades: int
    winning_trades: int
    losing_trades: int
    avg_win_pct: float
    avg_loss_pct: float
    profit_factor: float
    trades: list[Trade] = field(default_factory=list)
    equity_curve: list[dict] = field(default_factory=list)

    def summary(self) -> str:
        """Return a formatted summary."""
        return f"""
╔══════════════════════════════════════════╗
║       NeuralTrade Backtest Results       ║
╠══════════════════════════════════════════╣
║ Symbol:          {self.symbol:<25} ║
║ Period:          {self.start_date} → {self.end_date} ║
║ Initial Capital: ${self.initial_capital:>12,.2f}          ║
║ Final Value:     ${self.final_value:>12,.2f}          ║
║ Total Return:    {self.total_return_pct:>10.2f}%              ║
║ Annual Return:   {self.annualized_return_pct:>10.2f}%              ║
║ Sharpe Ratio:    {self.sharpe_ratio:>10.2f}               ║
║ Max Drawdown:    {self.max_drawdown_pct:>10.2f}%              ║
║ Win Rate:        {self.win_rate:>10.1f}%              ║
║ Total Trades:    {self.total_trades:>10}               ║
║ Profit Factor:   {self.profit_factor:>10.2f}               ║
╚══════════════════════════════════════════╝
"""


class BacktestEngine:
    """Backtesting engine for trading strategies."""

    def __init__(
        self,
        initial_capital: float = 100_000.0,
        commission: float = 0.001,  # 0.1% per trade
        slippage: float = 0.0005,  # 0.05% slippage
    ):
        self.initial_capital = initial_capital
        self.commission = commission
        self.slippage = slippage

    def run(
        self,
        symbol: str,
        price_data: pd.DataFrame,
        signals: pd.Series,  # 1 = BUY, -1 = SELL, 0 = HOLD
    ) -> BacktestResult:
        """Run backtest on historical data with given signals."""
        if price_data.empty or signals.empty:
            return self._empty_result(symbol)

        capital = self.initial_capital
        position = 0.0
        trades: list[Trade] = []
        equity_curve: list[dict] = []
        peak_value = self.initial_capital
        max_drawdown = 0.0

        dates = price_data.index
        closes = price_data["Close"]

        for i, date in enumerate(dates):
            price = closes.iloc[i]
            signal = signals.iloc[i] if i < len(signals) else 0

            # Apply slippage
            if signal == 1:  # BUY
                exec_price = price * (1 + self.slippage)
                max_shares = int(capital / exec_price)
                if max_shares > 0:
                    cost = max_shares * exec_price * (1 + self.commission)
                    capital -= cost
                    position += max_shares
                    trades.append(Trade(
                        date=str(date.date()), symbol=symbol,
                        action="BUY", price=exec_price,
                        quantity=max_shares, value=cost,
                    ))
            elif signal == -1 and position > 0:  # SELL
                exec_price = price * (1 - self.slippage)
                revenue = position * exec_price * (1 - self.commission)
                capital += revenue
                trades.append(Trade(
                    date=str(date.date()), symbol=symbol,
                    action="SELL", price=exec_price,
                    quantity=position, value=revenue,
                ))
                position = 0

            # Track equity
            total_value = capital + (position * price)
            equity_curve.append({"date": str(date.date()), "value": total_value})

            # Track drawdown
            if total_value > peak_value:
                peak_value = total_value
            dd = (peak_value - total_value) / peak_value * 100
            if dd > max_drawdown:
                max_drawdown = dd

        # Final value
        final_price = closes.iloc[-1]
        final_value = capital + (position * final_price)
        total_return = (final_value - self.initial_capital) / self.initial_capital * 100

        # Calculate metrics
        winning = [t for t in trades if t.action == "SELL" and t.value > 0]
        losing = [t for t in trades if t.action == "SELL" and t.value <= 0]

        # Sharpe ratio (simplified)
        if len(equity_curve) > 1:
            values = [e["value"] for e in equity_curve]
            returns = np.diff(values) / values[:-1]
            sharpe = np.mean(returns) / np.std(returns) * np.sqrt(252) if np.std(returns) > 0 else 0
        else:
            sharpe = 0

        # Annualized return
        days = (dates[-1] - dates[0]).days if len(dates) > 1 else 1
        years = max(days / 365.25, 0.01)
        annual_return = ((final_value / self.initial_capital) ** (1 / years) - 1) * 100

        sell_trades = [t for t in trades if t.action == "SELL"]
        win_rate = (len(winning) / len(sell_trades) * 100) if sell_trades else 0

        return BacktestResult(
            symbol=symbol,
            start_date=str(dates[0].date()),
            end_date=str(dates[-1].date()),
            initial_capital=self.initial_capital,
            final_value=round(final_value, 2),
            total_return_pct=round(total_return, 2),
            annualized_return_pct=round(annual_return, 2),
            sharpe_ratio=round(sharpe, 2),
            max_drawdown_pct=round(max_drawdown, 2),
            win_rate=round(win_rate, 1),
            total_trades=len(trades),
            winning_trades=len(winning),
            losing_trades=len(losing),
            avg_win_pct=0.0,
            avg_loss_pct=0.0,
            profit_factor=0.0,
            trades=trades,
            equity_curve=equity_curve,
        )

    def _empty_result(self, symbol: str) -> BacktestResult:
        return BacktestResult(
            symbol=symbol, start_date="", end_date="",
            initial_capital=self.initial_capital, final_value=self.initial_capital,
            total_return_pct=0, annualized_return_pct=0, sharpe_ratio=0,
            max_drawdown_pct=0, win_rate=0, total_trades=0,
            winning_trades=0, losing_trades=0,
            avg_win_pct=0, avg_loss_pct=0, profit_factor=0,
        )
