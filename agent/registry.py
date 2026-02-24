"""
registry.py — ERC-8004 on-chain registry interactions.

Handles identity registration, reputation feedback submission,
and validation request/response lifecycle.
"""

import json
import os
from pathlib import Path
from typing import Optional

from web3 import Web3
from web3.contract import Contract
from eth_account import Account
from loguru import logger


def load_abi(name: str) -> list:
    """Load contract ABI from the compiled artifacts."""
    artifacts_dir = Path(__file__).parent.parent / "contracts" / "artifacts" / "contracts"
    abi_path = artifacts_dir / f"{name}.sol" / f"{name}.json"
    if not abi_path.exists():
        raise FileNotFoundError(f"ABI not found: {abi_path}. Run 'npx hardhat compile' first.")
    with open(abi_path) as f:
        return json.load(f)["abi"]


class ERC8004Registry:
    """
    Interface to all ERC-8004 contracts.

    Usage:
        reg = ERC8004Registry(w3, account)
        agent_id = reg.register_agent("ipfs://Qm...")
        reg.submit_validation_request(agent_id, data_uri, data_hash)
    """

    def __init__(self, w3: Web3, account: Account):
        self.w3 = w3
        self.account = account

        # Load contract addresses from env
        self.identity_addr = os.getenv("IDENTITY_REGISTRY_ADDRESS")
        self.reputation_addr = os.getenv("REPUTATION_REGISTRY_ADDRESS")
        self.validation_addr = os.getenv("VALIDATION_REGISTRY_ADDRESS")
        self.wallet_addr = os.getenv("AGENT_WALLET_ADDRESS")

        self._identity: Optional[Contract] = None
        self._reputation: Optional[Contract] = None
        self._validation: Optional[Contract] = None

    def _get_identity(self) -> Contract:
        if not self._identity:
            abi = load_abi("IdentityRegistry")
            self._identity = self.w3.eth.contract(
                address=self.identity_addr, abi=abi
            )
        return self._identity

    def _get_reputation(self) -> Contract:
        if not self._reputation:
            abi = load_abi("ReputationRegistry")
            self._reputation = self.w3.eth.contract(
                address=self.reputation_addr, abi=abi
            )
        return self._reputation

    def _get_validation(self) -> Contract:
        if not self._validation:
            abi = load_abi("ValidationRegistry")
            self._validation = self.w3.eth.contract(
                address=self.validation_addr, abi=abi
            )
        return self._validation

    def _send_tx(self, fn, gas: int = 200_000) -> str:
        """Build, sign, and send a transaction. Returns tx hash."""
        nonce = self.w3.eth.get_transaction_count(self.account.address)
        tx = fn.build_transaction({
            "from": self.account.address,
            "nonce": nonce,
            "gas": gas,
            "gasPrice": self.w3.eth.gas_price,
        })
        signed = self.account.sign_transaction(tx)
        tx_hash = self.w3.eth.send_raw_transaction(signed.raw_transaction)
        receipt = self.w3.eth.wait_for_transaction_receipt(tx_hash, timeout=120)
        logger.info(f"TX confirmed: {tx_hash.hex()} (block {receipt['blockNumber']})")
        return tx_hash.hex()

    # ─── Identity ────────────────────────────────────────────────────────────

    def get_agent_id(self) -> int:
        """Return the agent ID for this account, or 0 if not registered."""
        registry = self._get_identity()
        return registry.functions.agentIdOf(self.account.address).call()

    def is_registered(self) -> bool:
        registry = self._get_identity()
        return registry.functions.isRegistered(self.account.address).call()

    def register_agent(self, metadata_uri: str) -> int:
        """
        Register this agent on-chain. Returns the assigned agent ID.
        Skips if already registered.
        """
        if self.is_registered():
            agent_id = self.get_agent_id()
            logger.info(f"Agent already registered with ID: {agent_id}")
            return agent_id

        logger.info(f"Registering agent with URI: {metadata_uri}")
        registry = self._get_identity()
        fn = registry.functions.mint(metadata_uri)
        self._send_tx(fn)

        agent_id = self.get_agent_id()
        logger.success(f"Agent registered! ID: {agent_id}")
        return agent_id

    def get_agent_did(self, agent_id: int) -> str:
        registry = self._get_identity()
        return registry.functions.agentDID(agent_id).call()

    def update_agent_uri(self, agent_id: int, new_uri: str) -> str:
        registry = self._get_identity()
        fn = registry.functions.setAgentURI(agent_id, new_uri)
        return self._send_tx(fn)

    # ─── Reputation ──────────────────────────────────────────────────────────

    def get_aggregate_score(self, agent_id: int) -> tuple[int, int]:
        """Returns (score, count). Score is in 2 decimal fixed-point (divide by 100)."""
        registry = self._get_reputation()
        return registry.functions.getAggregateScore(agent_id).call()

    def get_feedback_count(self, agent_id: int) -> int:
        registry = self._get_reputation()
        return registry.functions.feedbackCount(agent_id).call()

    def give_feedback(
        self,
        agent_id: int,
        score: int,           # e.g. 850 (= 8.50 with decimals=2)
        decimals: int = 2,
        tag1: str = "accuracy",
        tag2: str = "trading",
        endpoint_uri: str = "",
        file_hash: bytes = b"\x00" * 32,
    ) -> str:
        """Submit performance feedback for an agent."""
        tag1_bytes = tag1.encode().ljust(32, b"\x00")[:32]
        tag2_bytes = tag2.encode().ljust(32, b"\x00")[:32]

        registry = self._get_reputation()
        fn = registry.functions.giveFeedback(
            agent_id,
            score,
            decimals,
            tag1_bytes,
            tag2_bytes,
            endpoint_uri,
            file_hash,
        )
        return self._send_tx(fn)

    # ─── Validation ──────────────────────────────────────────────────────────

    def request_validation(
        self,
        agent_id: int,
        data_uri: str,
        data_hash: bytes,
    ) -> tuple[str, int]:
        """
        Submit a validation request for a trade or action.
        Returns (tx_hash, request_id).
        """
        registry = self._get_validation()
        fn = registry.functions.validationRequest(agent_id, data_uri, data_hash)

        # Get the request ID from the event log
        nonce = self.w3.eth.get_transaction_count(self.account.address)
        tx = fn.build_transaction({
            "from": self.account.address,
            "nonce": nonce,
            "gas": 200_000,
            "gasPrice": self.w3.eth.gas_price,
        })
        signed = self.account.sign_transaction(tx)
        tx_hash = self.w3.eth.send_raw_transaction(signed.raw_transaction)
        receipt = self.w3.eth.wait_for_transaction_receipt(tx_hash, timeout=120)

        # Decode ValidationRequested event to get requestId
        events = registry.events.ValidationRequested().process_receipt(receipt)
        request_id = events[0]["args"]["requestId"] if events else 0

        logger.info(f"Validation requested: requestId={request_id}, tx={tx_hash.hex()}")
        return tx_hash.hex(), request_id

    def submit_validation(
        self,
        request_id: int,
        response: int,       # 0-100
        comment_uri: str = "",
    ) -> str:
        """Submit a validation response (validator role)."""
        registry = self._get_validation()
        fn = registry.functions.submitValidation(request_id, response, comment_uri)
        return self._send_tx(fn)

    def get_validation_result(self, request_id: int) -> dict:
        registry = self._get_validation()
        validator, response, timestamp = registry.functions.getValidationResult(request_id).call()
        return {
            "validator": validator,
            "response": response,
            "timestamp": timestamp,
        }

    def get_agent_requests(self, agent_id: int) -> list[int]:
        registry = self._get_validation()
        return registry.functions.getAgentRequests(agent_id).call()
