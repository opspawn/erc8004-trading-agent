#!/usr/bin/env python3
"""
main.py — ERC-8004 Trading Agent Entry Point

Autonomous trading agent that:
1. Registers its identity on ERC-8004 IdentityRegistry
2. Scans prediction markets for trading opportunities
3. Executes trades (or simulates in dry-run mode)
4. Submits validation requests for each trade
5. Accumulates on-chain reputation

Usage:
    python main.py [--dry-run] [--once]

Environment:
    See .env.example for required configuration.
"""

import argparse
import asyncio
import os
import sys
from datetime import datetime, timezone

from dotenv import load_dotenv
from eth_account import Account
from loguru import logger
from web3 import Web3

from registry import ERC8004Registry
from trader import Market, TradingStrategy
from validator import TradeValidator

load_dotenv()


def setup_logging(log_level: str = "INFO") -> None:
    logger.remove()
    logger.add(
        sys.stdout,
        format="<green>{time:HH:mm:ss}</green> | <level>{level: <8}</level> | {message}",
        level=log_level,
        colorize=True,
    )
    logger.add(
        "logs/agent.log",
        rotation="10 MB",
        retention="7 days",
        level="DEBUG",
        format="{time:YYYY-MM-DD HH:mm:ss} | {level} | {message}",
    )


def connect_web3() -> Web3:
    rpc_url = os.getenv("RPC_URL", "http://localhost:8545")
    w3 = Web3(Web3.HTTPProvider(rpc_url))
    if not w3.is_connected():
        raise RuntimeError(f"Cannot connect to RPC: {rpc_url}")
    logger.info(f"Connected to {rpc_url} (chainId={w3.eth.chain_id})")
    return w3


def load_account() -> Account:
    private_key = os.getenv("AGENT_PRIVATE_KEY")
    if not private_key:
        raise ValueError("AGENT_PRIVATE_KEY not set in environment")
    account = Account.from_key(private_key)
    logger.info(f"Agent address: {account.address}")
    return account


def get_sample_markets() -> list[Market]:
    """
    Return sample markets for testing.
    In production: fetch from Polymarket/Manifold API.
    """
    return [
        Market(
            id="market-001",
            question="Will Bitcoin exceed $100k by March 31, 2026?",
            end_date="2026-03-31",
            yes_price=0.38,
            no_price=0.62,
            volume=250_000,
            category="crypto",
        ),
        Market(
            id="market-002",
            question="Will the US Federal Reserve cut rates in March 2026?",
            end_date="2026-03-20",
            yes_price=0.72,
            no_price=0.28,
            volume=180_000,
            category="macro",
        ),
        Market(
            id="market-003",
            question="Will SpaceX successfully land Starship in 2026?",
            end_date="2026-12-31",
            yes_price=0.65,
            no_price=0.35,
            volume=95_000,
            category="tech",
        ),
    ]


class TradingAgent:
    """
    Main agent loop.
    Coordinates identity registration, market scanning, trading, and validation.
    """

    SCAN_INTERVAL_SECONDS = 300  # 5 minutes between market scans

    def __init__(
        self,
        w3: Web3,
        account: Account,
        dry_run: bool = True,
    ):
        self.w3 = w3
        self.account = account
        self.dry_run = dry_run

        self.registry = ERC8004Registry(w3, account)
        self.strategy = TradingStrategy(dry_run=dry_run)
        self.validator = TradeValidator(registry=None if dry_run else self.registry)

        self.agent_id: int = 0
        self.cycle = 0

    async def startup(self) -> None:
        """Initialize the agent on-chain."""
        logger.info("=== ERC-8004 Trading Agent Starting ===")
        logger.info(f"Mode: {'DRY RUN' if self.dry_run else 'LIVE TRADING'}")
        logger.info(f"Address: {self.account.address}")

        metadata_uri = os.getenv(
            "AGENT_METADATA_URI",
            "ipfs://QmDefaultAgentMetadata"
        )

        if not self.dry_run:
            try:
                self.agent_id = self.registry.register_agent(metadata_uri)
                did = self.registry.get_agent_did(self.agent_id)
                logger.info(f"Agent DID: {did}")
            except Exception as e:
                logger.warning(f"Registry unavailable (contracts not deployed?): {e}")
                logger.info("Continuing without on-chain registration")
        else:
            logger.info("[DRY RUN] Skipping on-chain registration")
            self.agent_id = 1  # Mock agent ID

    async def scan_markets(self) -> list[Market]:
        """Scan for trading opportunities."""
        # TODO: Replace with Polymarket/Manifold API call
        markets = get_sample_markets()
        logger.info(f"Scanned {len(markets)} markets")
        return markets

    async def trading_cycle(self) -> None:
        """One full trading cycle: scan → decide → execute → validate."""
        self.cycle += 1
        now = datetime.now(timezone.utc).isoformat()
        logger.info(f"\n--- Cycle {self.cycle} | {now} ---")

        # 1. Scan markets
        markets = await self.scan_markets()

        # 2. Evaluate each market for opportunities
        decisions = []
        for market in markets:
            decision = self.strategy.evaluate_market(market)
            if decision:
                decisions.append(decision)
                logger.info(
                    f"SIGNAL: {decision.side} on '{market.question[:50]}...' "
                    f"(edge={decision.confidence:.2f})"
                )

        if not decisions:
            logger.info("No trading opportunities found this cycle")
            return

        # 3. Execute trades
        for decision in decisions:
            result = await self.strategy.execute_trade(decision)

            # 4. Submit validation request
            if not self.dry_run and self.agent_id > 0:
                tx_hash, request_id = self.validator.submit_to_chain(
                    self.agent_id, result
                )
                if tx_hash:
                    logger.info(f"Validation request submitted: {request_id}")
            else:
                artifact = self.validator.create_artifact(result)
                data_hash = self.validator.compute_data_hash(artifact)
                logger.info(
                    f"[DRY RUN] Validation artifact: checksum={artifact.checksum[:16]}..."
                )

        # 5. Performance summary
        summary = self.strategy.get_performance_summary()
        logger.info(
            f"Performance: {summary['total_trades']} trades, "
            f"PnL=${summary['total_pnl_usdc']:.2f}"
        )

    async def run(self, once: bool = False) -> None:
        """Main agent loop."""
        await self.startup()

        if once:
            await self.trading_cycle()
        else:
            while True:
                try:
                    await self.trading_cycle()
                except KeyboardInterrupt:
                    logger.info("Agent shutting down...")
                    break
                except Exception as e:
                    logger.error(f"Cycle error: {e}")

                logger.info(f"Sleeping {self.SCAN_INTERVAL_SECONDS}s until next cycle...")
                await asyncio.sleep(self.SCAN_INTERVAL_SECONDS)


async def main() -> None:
    parser = argparse.ArgumentParser(description="ERC-8004 Trading Agent")
    parser.add_argument("--dry-run", action="store_true", default=True,
                        help="Simulate trades without executing (default: True)")
    parser.add_argument("--live", action="store_true",
                        help="Enable live trading (overrides --dry-run)")
    parser.add_argument("--once", action="store_true",
                        help="Run one cycle then exit")
    args = parser.parse_args()

    dry_run = not args.live

    setup_logging(os.getenv("LOG_LEVEL", "INFO"))

    # Create logs directory
    os.makedirs("logs", exist_ok=True)

    try:
        w3 = connect_web3()
        account = load_account()
    except (RuntimeError, ValueError) as e:
        logger.error(f"Startup failed: {e}")
        sys.exit(1)

    agent = TradingAgent(w3, account, dry_run=dry_run)
    await agent.run(once=args.once)


if __name__ == "__main__":
    asyncio.run(main())
