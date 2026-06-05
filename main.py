"""Main entry point for NeuralTrade."""

from __future__ import annotations

import asyncio
import sys
from neuraltrade.cli import app as cli_app


def main():
    """Run the NeuralTrade CLI."""
    cli_app()


if __name__ == "__main__":
    main()
