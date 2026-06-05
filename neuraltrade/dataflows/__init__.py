"""Market data flows — stocks, crypto, news, and sentiment."""

from __future__ import annotations

import asyncio
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Any

import yfinance as yf
import pandas as pd
import numpy as np


@dataclass
class MarketData:
    """Container for market data."""
    symbol: str
    asset_type: str  # stock or crypto
    price: float = 0.0
    open: float = 0.0
    high: float = 0.0
    low: float = 0.0
    volume: float = 0.0
    change_pct: float = 0.0
    market_cap: float = 0.0
    pe_ratio: float = 0.0
    eps: float = 0.0
    dividend_yield: float = 0.0
    week_52_high: float = 0.0
    week_52_low: float = 0.0
    avg_volume: float = 0.0
    beta: float = 0.0
    history: pd.DataFrame = field(default_factory=pd.DataFrame)
    timestamp: str = ""


@dataclass
class TechnicalIndicators:
    """Technical analysis indicators."""
    sma_20: float = 0.0
    sma_50: float = 0.0
    sma_200: float = 0.0
    ema_12: float = 0.0
    ema_26: float = 0.0
    rsi_14: float = 0.0
    macd: float = 0.0
    macd_signal: float = 0.0
    macd_histogram: float = 0.0
    bollinger_upper: float = 0.0
    bollinger_lower: float = 0.0
    bollinger_middle: float = 0.0
    support: float = 0.0
    resistance: float = 0.0
    trend: str = "neutral"  # bullish, bearish, neutral


@dataclass
class NewsItem:
    """A single news item."""
    title: str
    source: str
    url: str
    published: str
    summary: str = ""
    sentiment: str = "neutral"  # positive, negative, neutral


@dataclass
class SentimentData:
    """Aggregated sentiment data."""
    overall: str = "neutral"  # bullish, bearish, neutral
    score: float = 0.0  # -1 to 1
    news_sentiment: float = 0.0
    social_sentiment: float = 0.0
    analyst_rating: str = ""
    news_items: list[NewsItem] = field(default_factory=list)


class MarketDataFlow:
    """Fetches and processes market data."""

    def __init__(self, cache_dir: str = ".cache"):
        self.cache_dir = cache_dir

    async def get_market_data(self, symbol: str, period: str = "3mo") -> MarketData:
        """Fetch market data for a symbol."""
        try:
            ticker = yf.Ticker(symbol)
            info = ticker.info
            hist = ticker.history(period=period)

            if hist.empty:
                return MarketData(symbol=symbol, asset_type="stock")

            latest = hist.iloc[-1]
            prev = hist.iloc[-2] if len(hist) > 1 else latest

            change_pct = ((latest["Close"] - prev["Close"]) / prev["Close"]) * 100 if prev["Close"] > 0 else 0

            return MarketData(
                symbol=symbol,
                asset_type="stock",
                price=latest["Close"],
                open=latest["Open"],
                high=latest["High"],
                low=latest["Low"],
                volume=latest["Volume"],
                change_pct=round(change_pct, 2),
                market_cap=info.get("marketCap", 0),
                pe_ratio=info.get("trailingPE", 0),
                eps=info.get("trailingEps", 0),
                dividend_yield=info.get("dividendYield", 0),
                week_52_high=info.get("fiftyTwoWeekHigh", 0),
                week_52_low=info.get("fiftyTwoWeekLow", 0),
                avg_volume=info.get("averageVolume", 0),
                beta=info.get("beta", 0),
                history=hist,
                timestamp=datetime.now().isoformat(),
            )
        except Exception as e:
            return MarketData(symbol=symbol, asset_type="stock")

    async def get_technical_indicators(self, symbol: str, period: str = "6mo") -> TechnicalIndicators:
        """Calculate technical indicators."""
        try:
            ticker = yf.Ticker(symbol)
            hist = ticker.history(period=period)

            if hist.empty or len(hist) < 200:
                return TechnicalIndicators()

            close = hist["Close"]

            # SMAs
            sma_20 = close.rolling(20).mean().iloc[-1]
            sma_50 = close.rolling(50).mean().iloc[-1]
            sma_200 = close.rolling(200).mean().iloc[-1]

            # EMAs
            ema_12 = close.ewm(span=12).mean().iloc[-1]
            ema_26 = close.ewm(span=26).mean().iloc[-1]

            # MACD
            macd = ema_12 - ema_26
            macd_signal = close.ewm(span=9).mean().iloc[-1] - close.ewm(span=26).mean().iloc[-1]
            macd_histogram = macd - macd_signal

            # RSI
            delta = close.diff()
            gain = delta.where(delta > 0, 0).rolling(14).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
            rs = gain / loss
            rsi = 100 - (100 / (1 + rs))
            rsi_14 = rsi.iloc[-1]

            # Bollinger Bands
            bb_middle = close.rolling(20).mean().iloc[-1]
            bb_std = close.rolling(20).std().iloc[-1]
            bb_upper = bb_middle + (bb_std * 2)
            bb_lower = bb_middle - (bb_std * 2)

            # Support & Resistance (simplified)
            recent = close.tail(30)
            support = recent.min()
            resistance = recent.max()

            # Trend
            if sma_20 > sma_50 > sma_200:
                trend = "bullish"
            elif sma_20 < sma_50 < sma_200:
                trend = "bearish"
            else:
                trend = "neutral"

            return TechnicalIndicators(
                sma_20=round(sma_20, 2),
                sma_50=round(sma_50, 2),
                sma_200=round(sma_200, 2),
                ema_12=round(ema_12, 2),
                ema_26=round(ema_26, 2),
                rsi_14=round(rsi_14, 2),
                macd=round(macd, 4),
                macd_signal=round(macd_signal, 4),
                macd_histogram=round(macd_histogram, 4),
                bollinger_upper=round(bb_upper, 2),
                bollinger_lower=round(bb_lower, 2),
                bollinger_middle=round(bb_middle, 2),
                support=round(support, 2),
                resistance=round(resistance, 2),
                trend=trend,
            )
        except Exception:
            return TechnicalIndicators()


class NewsDataFlow:
    """Fetches news and sentiment data."""

    async def get_news(self, symbol: str, limit: int = 10) -> list[NewsItem]:
        """Fetch news for a symbol."""
        try:
            ticker = yf.Ticker(symbol)
            news = ticker.news
            items = []
            for item in news[:limit]:
                items.append(NewsItem(
                    title=item.get("title", ""),
                    source=item.get("publisher", ""),
                    url=item.get("link", ""),
                    published=item.get("published", ""),
                ))
            return items
        except Exception:
            return []

    async def get_sentiment(self, symbol: str) -> SentimentData:
        """Get aggregated sentiment for a symbol."""
        news_items = await self.get_news(symbol)

        # Simple sentiment from news titles
        positive_words = ["up", "rise", "gain", "bull", "growth", "beat", "surge", "rally", "strong", "buy"]
        negative_words = ["down", "fall", "loss", "bear", "drop", "miss", "crash", "weak", "sell", "debt"]

        pos_count = 0
        neg_count = 0
        for item in news_items:
            title_lower = item.title.lower()
            if any(w in title_lower for w in positive_words):
                pos_count += 1
                item.sentiment = "positive"
            elif any(w in title_lower for w in negative_words):
                neg_count += 1
                item.sentiment = "negative"

        total = pos_count + neg_count
        if total > 0:
            score = (pos_count - neg_count) / total
        else:
            score = 0

        if score > 0.2:
            overall = "bullish"
        elif score < -0.2:
            overall = "bearish"
        else:
            overall = "neutral"

        return SentimentData(
            overall=overall,
            score=round(score, 2),
            news_sentiment=round(score, 2),
            news_items=news_items,
        )
