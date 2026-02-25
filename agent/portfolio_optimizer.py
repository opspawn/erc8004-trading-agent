"""
portfolio_optimizer.py — Mean-variance portfolio optimizer for ERC-8004.

Implements:
  - Covariance-based mean-variance optimization (pure Python, no numpy dependency)
  - ERC-8004 reputation weights via Credora tier multipliers
  - Max position concentration per Credora tier
  - 5% drift rebalancing trigger
  - Integration with risk_manager.py validate_trade_with_sentiment()

Usage:
    opt = PortfolioOptimizer(credora_client=CredoraClient())
    returns = {"ETH": [0.01, -0.02, 0.005], "BTC": [0.008, -0.015, 0.012]}
    result = opt.optimize(returns)
    if opt.check_rebalance_needed(current_weights, result.weights):
        orders = opt.compute_rebalance_orders(current_weights, result.weights)
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

from loguru import logger


# ─── Max concentration limits by Credora tier ─────────────────────────────────

MAX_CONCENTRATION_BY_TIER: Dict[str, float] = {
    "AAA": 0.40,   # 40% max — highest trust, minimal concentration risk
    "AA":  0.35,
    "A":   0.30,
    "BBB": 0.25,
    "BB":  0.20,
    "B":   0.15,
    "CCC": 0.10,   # Near-distressed — very low concentration
    "NR":  0.05,   # Unrated — treat conservatively
}

REBALANCE_THRESHOLD: float = 0.05  # 5% drift triggers rebalance


# ─── Data Classes ─────────────────────────────────────────────────────────────

@dataclass
class OptimizationResult:
    """Result of a portfolio optimization run."""
    weights: Dict[str, float]           # protocol → optimal weight
    expected_return: float              # annualised expected return
    expected_variance: float            # portfolio variance
    sharpe_ratio: float                 # annualised Sharpe ratio
    credora_adjustments: Dict[str, float]  # protocol → Credora multiplier applied
    concentration_caps: Dict[str, float]   # protocol → max allowed concentration
    rebalance_needed: bool
    method: str = "mean_variance"

    def to_dict(self) -> dict:
        return {
            "weights": {k: round(v, 6) for k, v in self.weights.items()},
            "expected_return": round(self.expected_return, 6),
            "expected_variance": round(self.expected_variance, 6),
            "sharpe_ratio": round(self.sharpe_ratio, 6),
            "credora_adjustments": {k: round(v, 4) for k, v in self.credora_adjustments.items()},
            "concentration_caps": {k: round(v, 4) for k, v in self.concentration_caps.items()},
            "rebalance_needed": self.rebalance_needed,
            "method": self.method,
        }


@dataclass
class RebalanceOrder:
    """A single rebalancing trade order."""
    protocol: str
    current_weight: float
    target_weight: float
    drift: float       # current - target (positive = overweight)
    direction: str     # "BUY" or "SELL"
    urgency: str       # "high" (>10%), "medium" (5-10%), "low" (<5%)

    def to_dict(self) -> dict:
        return {
            "protocol": self.protocol,
            "current_weight": round(self.current_weight, 6),
            "target_weight": round(self.target_weight, 6),
            "drift": round(self.drift, 6),
            "direction": self.direction,
            "urgency": self.urgency,
        }


# ─── Portfolio Optimizer ───────────────────────────────────────────────────────

class PortfolioOptimizer:
    """
    Mean-variance portfolio optimizer with ERC-8004 reputation weighting.

    Integrates:
      - Credora ratings → position concentration caps + weighting multipliers
      - Pure Python covariance estimation (no numpy dependency)
      - Rebalancing logic: triggers when any position drifts > 5% from target
      - Risk manager integration for sentiment-adjusted trade validation

    Optimization methods:
      - "min_variance"  : minimize portfolio variance (GMV portfolio)
      - "max_sharpe"    : maximize Sharpe ratio (tangency portfolio)
      - "equal_weight"  : 1/N baseline

    The Credora tier acts as a reputation score that scales position weights:
      - AAA protocol can hold up to 40% of portfolio
      - NR (unrated) protocol capped at 5% — high due-diligence risk
    """

    def __init__(
        self,
        credora_client=None,
        risk_manager=None,
        rebalance_threshold: float = REBALANCE_THRESHOLD,
        risk_free_rate: float = 0.04,   # 4% annual risk-free rate
        max_weight: float = 0.40,       # hard cap when Credora not available
        min_weight: float = 0.0,        # allow exclusion (zero weight)
    ) -> None:
        self._credora = credora_client
        self._risk_manager = risk_manager
        self.rebalance_threshold = rebalance_threshold
        self.risk_free_rate = risk_free_rate
        self.max_weight = max_weight
        self.min_weight = min_weight

        logger.info(
            f"PortfolioOptimizer initialized: "
            f"rebalance_threshold={rebalance_threshold:.1%} "
            f"credora={'enabled' if credora_client else 'disabled'}"
        )

    # ─── Statistics Helpers (pure Python) ─────────────────────────────────────

    @staticmethod
    def _mean(xs: List[float]) -> float:
        if not xs:
            return 0.0
        return sum(xs) / len(xs)

    @staticmethod
    def _variance(xs: List[float]) -> float:
        if len(xs) < 2:
            return 1e-6
        m = PortfolioOptimizer._mean(xs)
        return sum((x - m) ** 2 for x in xs) / (len(xs) - 1)

    @staticmethod
    def _covariance(xs: List[float], ys: List[float]) -> float:
        n = min(len(xs), len(ys))
        if n < 2:
            return 0.0
        mx = PortfolioOptimizer._mean(xs[:n])
        my = PortfolioOptimizer._mean(ys[:n])
        return sum((xs[i] - mx) * (ys[i] - my) for i in range(n)) / (n - 1)

    def _build_cov_matrix(
        self, returns_dict: Dict[str, List[float]]
    ) -> Tuple[List[str], List[List[float]]]:
        """Build full covariance matrix from returns dict."""
        assets = sorted(returns_dict.keys())
        n = len(assets)
        cov: List[List[float]] = [[0.0] * n for _ in range(n)]
        for i, a in enumerate(assets):
            for j, b in enumerate(assets):
                if i == j:
                    cov[i][j] = self._variance(returns_dict[a])
                elif j > i:
                    c = self._covariance(returns_dict[a], returns_dict[b])
                    cov[i][j] = c
                    cov[j][i] = c
        return assets, cov

    def _portfolio_variance(
        self, weights: List[float], cov: List[List[float]]
    ) -> float:
        n = len(weights)
        return sum(
            weights[i] * weights[j] * cov[i][j]
            for i in range(n) for j in range(n)
        )

    def _portfolio_return(
        self, weights: List[float], mean_returns: List[float]
    ) -> float:
        return sum(w * r for w, r in zip(weights, mean_returns))

    # ─── Credora Integration ───────────────────────────────────────────────────

    def _get_credora_multiplier(self, protocol: str) -> float:
        """Return Credora Kelly multiplier for a protocol (0.10–1.00)."""
        if self._credora is None:
            return 1.0
        try:
            rating = self._credora.get_rating(protocol)
            return rating.kelly_multiplier
        except Exception:
            return 0.50  # conservative fallback

    def _get_max_concentration(self, protocol: str) -> float:
        """Return max weight cap for a protocol based on Credora tier."""
        if self._credora is None:
            return self.max_weight
        try:
            rating = self._credora.get_rating(protocol)
            return MAX_CONCENTRATION_BY_TIER.get(rating.tier.value, self.max_weight)
        except Exception:
            return 0.10  # conservative fallback

    # ─── Optimization ──────────────────────────────────────────────────────────

    def optimize(
        self,
        returns_dict: Dict[str, List[float]],
        method: str = "min_variance",
    ) -> OptimizationResult:
        """
        Optimize portfolio weights using mean-variance optimization.

        Steps:
          1. Compute covariance matrix from historical returns
          2. Find base weights (min variance or max Sharpe)
          3. Apply Credora multipliers to down-weight riskier protocols
          4. Enforce per-protocol concentration caps by Credora tier
          5. Normalize to sum = 1.0

        Args:
            returns_dict: {protocol: [daily_return, ...]}
            method: "min_variance", "max_sharpe", or "equal_weight"

        Returns:
            OptimizationResult with optimal weights and portfolio metrics
        """
        if not returns_dict:
            return OptimizationResult(
                weights={}, expected_return=0.0, expected_variance=0.0,
                sharpe_ratio=0.0, credora_adjustments={},
                concentration_caps={}, rebalance_needed=False, method=method,
            )

        assets, cov = self._build_cov_matrix(returns_dict)
        n = len(assets)
        mean_returns = [self._mean(returns_dict[a]) for a in assets]

        # Step 1: Base weights by method
        if method == "min_variance":
            weights = self._min_variance_weights(cov, n)
        elif method == "max_sharpe":
            weights = self._max_sharpe_weights(cov, mean_returns, n)
        elif method == "equal_weight":
            weights = [1.0 / n] * n
        else:
            raise ValueError(f"Unknown method {method!r}. Choose: min_variance, max_sharpe, equal_weight")

        # Step 2: Apply Credora multipliers (reputation-weighted scaling)
        credora_mults: Dict[str, float] = {a: self._get_credora_multiplier(a) for a in assets}
        for i, a in enumerate(assets):
            weights[i] *= credora_mults[a]

        # Step 3: Enforce concentration caps by Credora tier
        caps: Dict[str, float] = {a: self._get_max_concentration(a) for a in assets}

        # Step 4: Normalize, cap, renormalize
        weights = self._normalize_and_cap(weights, [caps[a] for a in assets])

        # Step 5: Compute portfolio metrics
        port_return = self._portfolio_return(weights, mean_returns)
        port_var = self._portfolio_variance(weights, cov)
        port_std = math.sqrt(max(port_var, 0.0))

        # Annualise (daily returns → annual)
        ann_return = port_return * 252
        ann_std = port_std * math.sqrt(252)

        if ann_std > 1e-9:
            sharpe = (ann_return - self.risk_free_rate) / ann_std
        else:
            sharpe = 0.0

        result_weights = {assets[i]: round(weights[i], 6) for i in range(n)}

        logger.info(
            f"PortfolioOptimizer: optimized {n} assets "
            f"method={method} sharpe={sharpe:.3f} var={port_var:.6f}"
        )

        return OptimizationResult(
            weights=result_weights,
            expected_return=ann_return,
            expected_variance=port_var,
            sharpe_ratio=sharpe,
            credora_adjustments=credora_mults,
            concentration_caps=caps,
            rebalance_needed=False,
            method=method,
        )

    def _min_variance_weights(
        self, cov: List[List[float]], n: int
    ) -> List[float]:
        """
        Approximate global minimum variance (GMV) weights.

        Uses inverse-variance weighting as starting point, then refines
        via iterative gradient descent on portfolio variance.
        """
        if n == 1:
            return [1.0]

        # Initial weights: inverse-variance (diagonal approximation)
        diag_var = [max(cov[i][i], 1e-9) for i in range(n)]
        inv_vars = [1.0 / v for v in diag_var]
        total = sum(inv_vars)
        weights = [iv / total for iv in inv_vars]

        # Refine with gradient descent (20 iterations typically sufficient)
        lr = 0.05
        for _ in range(30):
            port_var = self._portfolio_variance(weights, cov)
            # ∂σ²/∂wᵢ = 2 * Σⱼ wⱼ cov[i][j]
            gradients = [
                2 * sum(weights[j] * cov[i][j] for j in range(n))
                for i in range(n)
            ]
            new_weights = [max(0.0, weights[i] - lr * gradients[i]) for i in range(n)]
            s = sum(new_weights)
            if s <= 0:
                break
            new_weights = [w / s for w in new_weights]
            new_var = self._portfolio_variance(new_weights, cov)
            if new_var < port_var:
                weights = new_weights

        return weights

    def _max_sharpe_weights(
        self,
        cov: List[List[float]],
        mean_returns: List[float],
        n: int,
    ) -> List[float]:
        """
        Approximate maximum Sharpe ratio (tangency portfolio) weights.

        Iterative gradient ascent on Sharpe ratio. Starts from equal weights.
        """
        if n == 1:
            return [1.0]

        daily_rfr = self.risk_free_rate / 252.0
        excess = [r - daily_rfr for r in mean_returns]

        weights = [1.0 / n] * n
        best_sharpe = -float("inf")
        best_weights = weights[:]

        for _ in range(60):
            port_ret = self._portfolio_return(weights, excess)
            port_var = self._portfolio_variance(weights, cov)
            port_std = math.sqrt(max(port_var, 1e-12))
            sharpe = port_ret / port_std

            if sharpe > best_sharpe:
                best_sharpe = sharpe
                best_weights = weights[:]

            # Gradient of Sharpe w.r.t. weights (chain rule)
            var_grad = [
                2 * sum(weights[j] * cov[i][j] for j in range(n))
                for i in range(n)
            ]
            step = 0.03
            new_weights = [
                max(0.0, weights[i] + step * (
                    excess[i] / port_std - sharpe * var_grad[i] / (2 * port_std)
                ))
                for i in range(n)
            ]
            s = sum(new_weights)
            if s <= 0:
                break
            weights = [w / s for w in new_weights]

        return best_weights

    def _normalize_and_cap(
        self, weights: List[float], caps: List[float]
    ) -> List[float]:
        """Normalize weights to sum=1, enforce caps via iterative projection.

        If sum(caps) < 1.0, the problem is infeasible (caps are too tight).
        In that case, we scale the caps up proportionally so they sum to 1.0.
        """
        n = len(weights)
        if n == 0:
            return []

        weights = [max(0.0, w) for w in weights]

        # Feasibility check: if sum(caps) < 1, scale caps up proportionally
        cap_sum = sum(caps)
        if cap_sum < 1.0 - 1e-9:
            scale = 1.0 / cap_sum
            caps = [min(c * scale, 1.0) for c in caps]

        for _ in range(200):  # iterate until convergence
            total = sum(weights)
            if total <= 0:
                return [1.0 / n] * n
            weights = [w / total for w in weights]

            capped = False
            for i in range(n):
                if weights[i] > caps[i]:
                    weights[i] = caps[i]
                    capped = True

            if not capped:
                break

        # Final normalization
        total = sum(weights)
        if total <= 0:
            return [1.0 / n] * n
        return [w / total for w in weights]

    # ─── Rebalancing ──────────────────────────────────────────────────────────

    def check_rebalance_needed(
        self,
        current_weights: Dict[str, float],
        target_weights: Dict[str, float],
        threshold: Optional[float] = None,
    ) -> bool:
        """
        Check whether portfolio rebalancing is needed.

        Triggers when any protocol's weight drifts more than `threshold`
        from its target allocation. Default threshold: 5%.

        Args:
            current_weights: {protocol: current_fraction}
            target_weights:  {protocol: target_fraction}
            threshold:       Override rebalance_threshold (default: 5%)

        Returns:
            True if rebalancing is needed
        """
        thr = threshold if threshold is not None else self.rebalance_threshold
        all_protocols = set(current_weights) | set(target_weights)

        for protocol in all_protocols:
            current = current_weights.get(protocol, 0.0)
            target = target_weights.get(protocol, 0.0)
            drift = abs(current - target)
            if drift > thr:
                logger.info(
                    f"PortfolioOptimizer: rebalance triggered by {protocol!r} "
                    f"drift={drift:.2%} > threshold={thr:.2%}"
                )
                return True
        return False

    def compute_rebalance_orders(
        self,
        current_weights: Dict[str, float],
        target_weights: Dict[str, float],
        portfolio_value: float = 1.0,
    ) -> List[RebalanceOrder]:
        """
        Compute list of rebalancing trades to move from current to target weights.

        Args:
            current_weights: {protocol: current_fraction}
            target_weights:  {protocol: target_fraction}
            portfolio_value: Total portfolio value in USDC (for sizing)

        Returns:
            List of RebalanceOrder sorted by urgency (high first), then drift size
        """
        all_protocols = set(current_weights) | set(target_weights)
        orders: List[RebalanceOrder] = []

        for protocol in all_protocols:
            current = current_weights.get(protocol, 0.0)
            target = target_weights.get(protocol, 0.0)
            drift = current - target  # positive = overweight, needs SELL
            if abs(drift) < 0.001:
                continue

            direction = "SELL" if drift > 0 else "BUY"
            abs_drift = abs(drift)
            if abs_drift > 0.10:
                urgency = "high"
            elif abs_drift > 0.05:
                urgency = "medium"
            else:
                urgency = "low"

            orders.append(RebalanceOrder(
                protocol=protocol,
                current_weight=current,
                target_weight=target,
                drift=drift,
                direction=direction,
                urgency=urgency,
            ))

        # Sort: urgency (high → medium → low), then absolute drift descending
        urgency_rank = {"high": 0, "medium": 1, "low": 2}
        orders.sort(key=lambda o: (urgency_rank[o.urgency], -abs(o.drift)))
        return orders

    def validate_rebalance_with_risk_manager(
        self,
        order: RebalanceOrder,
        portfolio_value: float,
        price: float = 0.5,
        ticker: str = "",
    ) -> Tuple[bool, str, float]:
        """
        Validate a rebalancing order through RiskManager.validate_trade_with_sentiment().

        Args:
            order:           The rebalancing order to validate
            portfolio_value: Total portfolio value in USDC
            price:           Market price (prediction market 0–1)
            ticker:          Ticker for sentiment lookup (defaults to protocol name)

        Returns:
            (allowed: bool, reason: str, adjusted_size: float)
        """
        size = abs(order.drift) * portfolio_value
        ticker = ticker or order.protocol

        if self._risk_manager is None:
            return True, "no_risk_manager", size

        try:
            ok, reason, adj_size = self._risk_manager.validate_trade_with_sentiment(
                side=order.direction,
                size=size,
                price=price,
                portfolio_value=portfolio_value,
                ticker=ticker,
            )
            return ok, reason, adj_size
        except Exception as e:
            logger.warning(f"PortfolioOptimizer: risk manager validation failed: {e}")
            return True, f"risk_manager_error: {e}", size

    # ─── Drift Analysis ────────────────────────────────────────────────────────

    def compute_drift(
        self,
        current_weights: Dict[str, float],
        target_weights: Dict[str, float],
    ) -> Dict[str, float]:
        """
        Compute drift for each protocol (current - target).
        Positive = overweight, negative = underweight.
        """
        all_protocols = set(current_weights) | set(target_weights)
        return {
            p: current_weights.get(p, 0.0) - target_weights.get(p, 0.0)
            for p in all_protocols
        }

    def max_drift(
        self,
        current_weights: Dict[str, float],
        target_weights: Dict[str, float],
    ) -> float:
        """Return the maximum absolute drift across all protocols."""
        drift = self.compute_drift(current_weights, target_weights)
        if not drift:
            return 0.0
        return max(abs(d) for d in drift.values())

    def effective_weights_with_credora(
        self,
        raw_weights: Dict[str, float],
    ) -> Dict[str, float]:
        """
        Apply Credora multipliers to raw weights and renormalize.

        Lower-rated protocols receive reduced weight. Useful for post-hoc
        adjustment of user-supplied weights without full re-optimization.
        """
        if not raw_weights:
            return {}
        adjusted = {
            protocol: w * self._get_credora_multiplier(protocol)
            for protocol, w in raw_weights.items()
        }
        total = sum(adjusted.values())
        if total <= 0:
            eq = 1.0 / len(raw_weights)
            return {p: eq for p in raw_weights}
        return {p: w / total for p, w in adjusted.items()}

    def portfolio_risk_summary(
        self,
        weights: Dict[str, float],
        returns_dict: Dict[str, List[float]],
    ) -> dict:
        """
        Compute risk summary for a given weight allocation.

        Returns expected return, variance, Sharpe, and per-asset contributions.
        """
        if not weights or not returns_dict:
            return {}

        assets = sorted(weights.keys())
        _, cov = self._build_cov_matrix(returns_dict)
        mean_returns = [self._mean(returns_dict.get(a, [0.0])) for a in assets]
        w_list = [weights.get(a, 0.0) for a in assets]

        port_ret = self._portfolio_return(w_list, mean_returns)
        port_var = self._portfolio_variance(w_list, cov)
        port_std = math.sqrt(max(port_var, 0.0))

        ann_ret = port_ret * 252
        ann_std = port_std * math.sqrt(252)
        sharpe = (ann_ret - self.risk_free_rate) / ann_std if ann_std > 1e-9 else 0.0

        return {
            "expected_annual_return": round(ann_ret, 6),
            "expected_variance": round(port_var, 6),
            "expected_annual_std": round(ann_std, 6),
            "sharpe_ratio": round(sharpe, 6),
            "per_asset": {
                a: {
                    "weight": round(weights.get(a, 0.0), 6),
                    "mean_daily_return": round(mean_returns[i], 6),
                    "variance": round(cov[i][i], 6),
                }
                for i, a in enumerate(assets)
            },
        }
