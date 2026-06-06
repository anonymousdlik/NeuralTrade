"""NeuralTrade configuration management."""

from __future__ import annotations

import os
from dataclasses import dataclass, field

from dotenv import load_dotenv

load_dotenv()


@dataclass
class ModelConfig:
    """Model routing configuration."""
    default_model: str = "openai/gpt-4o-mini"
    reasoning_model: str = "openai/gpt-4o"
    fast_model: str = "openai/gpt-4o-mini"
    # Per-agent overrides
    analyst_model: str = ""
    researcher_model: str = ""
    trader_model: str = ""
    risk_manager_model: str = ""
    portfolio_manager_model: str = ""

    def get_model(self, agent: str | None = None) -> str:
        """Get model for a specific agent or default."""
        if agent:
            override = getattr(self, f"{agent}_model", "")
            if override:
                return override
        return self.default_model


@dataclass
class TradingConfig:
    """Trading parameters."""
    initial_capital: float = 100_000.0
    max_position_size: float = 0.25  # Max 25% of portfolio per position
    stop_loss_pct: float = 0.05
    take_profit_pct: float = 0.10
    paper_trading: bool = True
    max_daily_trades: int = 10


@dataclass
class DataConfig:
    """Market data configuration."""
    alpha_vantage_key: str = ""
    news_api_key: str = ""
    default_period: str = "3mo"
    cache_dir: str = ".cache"


@dataclass
class ServerConfig:
    """Dashboard server configuration."""
    host: str = "0.0.0.0"
    port: int = 8080


@dataclass
class AppConfig:
    """Main application configuration."""
    model: ModelConfig = field(default_factory=ModelConfig)
    trading: TradingConfig = field(default_factory=TradingConfig)
    data: DataConfig = field(default_factory=DataConfig)
    server: ServerConfig = field(default_factory=ServerConfig)

    # API keys
    openai_key: str = ""
    anthropic_key: str = ""
    google_key: str = ""
    openrouter_key: str = ""
    groq_key: str = ""
    deepseek_key: str = ""

    @classmethod
    def from_env(cls) -> AppConfig:
        """Load configuration from environment variables."""
        return cls(
            model=ModelConfig(
                default_model=os.getenv("NEURALTRADE_DEFAULT_MODEL", "openai/gpt-4o-mini"),
                reasoning_model=os.getenv("NEURALTRADE_REASONING_MODEL", "openai/gpt-4o"),
                fast_model=os.getenv("NEURALTRADE_FAST_MODEL", "openai/gpt-4o-mini"),
                analyst_model=os.getenv("ANALYST_MODEL", ""),
                researcher_model=os.getenv("RESEARCHER_MODEL", ""),
                trader_model=os.getenv("TRADER_MODEL", ""),
                risk_manager_model=os.getenv("RISK_MANAGER_MODEL", ""),
                portfolio_manager_model=os.getenv("PORTFOLIO_MANAGER_MODEL", ""),
            ),
            trading=TradingConfig(
                initial_capital=float(os.getenv("INITIAL_CAPITAL", "100000")),
                max_position_size=float(os.getenv("MAX_POSITION_SIZE", "0.25")),
                stop_loss_pct=float(os.getenv("STOP_LOSS_PCT", "0.05")),
                take_profit_pct=float(os.getenv("TAKE_PROFIT_PCT", "0.10")),
                paper_trading=os.getenv("PAPER_TRADING", "true").lower() == "true",
            ),
            data=DataConfig(
                alpha_vantage_key=os.getenv("ALPHA_VANTAGE_API_KEY", ""),
                news_api_key=os.getenv("NEWS_API_KEY", ""),
            ),
            server=ServerConfig(
                host=os.getenv("NEURALTRADE_HOST", "0.0.0.0"),
                port=int(os.getenv("NEURALTRADE_PORT", "8080")),
            ),
            openai_key=os.getenv("OPENAI_API_KEY", ""),
            anthropic_key=os.getenv("ANTHROPIC_API_KEY", ""),
            google_key=os.getenv("GOOGLE_API_KEY", ""),
            openrouter_key=os.getenv("OPENROUTER_API_KEY", ""),
            groq_key=os.getenv("GROQ_API_KEY", ""),
            deepseek_key=os.getenv("DEEPSEEK_API_KEY", ""),
        )

    def has_key(self, provider: str) -> bool:
        """Check if API key is available for a provider."""
        key_map = {
            "openai": self.openai_key,
            "anthropic": self.anthropic_key,
            "google": self.google_key,
            "openrouter": self.openrouter_key,
            "groq": self.groq_key,
            "deepseek": self.deepseek_key,
        }
        return bool(key_map.get(provider, ""))
