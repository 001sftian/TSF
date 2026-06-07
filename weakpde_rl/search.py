from __future__ import annotations

from dataclasses import dataclass
import itertools
import random

from .linalg import column_norms, matvec, norm, ridge, select_columns
from .weakform import WeakSystem


@dataclass(frozen=True)
class DiscoveryResult:
    support: tuple[int, ...]
    coefficients: dict[str, float]
    relative_residual: float
    history: list[dict[str, object]]


class RNSPDEDiscoverer:
    """Weak-form RL/CEM support-set search for sparse PDE discovery."""

    def __init__(self, max_terms: int = 5, top_k: int = 10, ridge_lambda: float = 1e-6, cem_iterations: int = 8, population: int = 80, elite_fraction: float = 0.2, random_state: int = 3, sparsity_penalty: float = 2e-3, complexity_penalty: float = 1e-3):
        self.max_terms = max_terms
        self.top_k = top_k
        self.ridge_lambda = ridge_lambda
        self.cem_iterations = cem_iterations
        self.population = population
        self.elite_fraction = elite_fraction
        self.random_state = random_state
        self.sparsity_penalty = sparsity_penalty
        self.complexity_penalty = complexity_penalty

    def _score(self, system: WeakSystem, support: tuple[int, ...]) -> tuple[float, float]:
        y_norm = norm(system.y) + 1e-12
        if not support:
            return -1.0, 1.0
        x = select_columns(system.theta, support)
        beta = ridge(x, system.y, self.ridge_lambda)
        pred = matvec(x, beta)
        residual = norm([yy - pp for yy, pp in zip(system.y, pred)]) / y_norm
        penalty = self.sparsity_penalty * len(support) + self.complexity_penalty * sum(system.complexities[i] for i in support)
        return -(residual + penalty), residual

    def _initial_actions(self, system: WeakSystem) -> list[int]:
        y_norm = norm(system.y) + 1e-12
        norms = column_norms(system.theta)
        ranked = []
        for j in range(len(system.names)):
            corr = abs(sum(row[j] * yy for row, yy in zip(system.theta, system.y))) / ((norms[j] + 1e-12) * y_norm)
            ranked.append((corr - 0.015 * system.complexities[j], j))
        return [j for _, j in sorted(ranked, reverse=True)[: self.top_k]]

    def fit(self, system: WeakSystem) -> DiscoveryResult:
        rng = random.Random(self.random_state)
        actions = self._initial_actions(system)
        probs = [min(0.6, self.max_terms / max(len(actions), 1))] * len(actions)
        history: list[dict[str, object]] = []
        best_support: tuple[int, ...] = tuple()
        best_reward = -1e100
        elite_count = max(2, int(self.population * self.elite_fraction))
        for iteration in range(self.cem_iterations):
            candidates = []
            for _ in range(self.population):
                selected = [a for a, p in zip(actions, probs) if rng.random() < p]
                if not selected:
                    selected = [rng.choice(actions)]
                support = tuple(sorted(selected[: self.max_terms]))
                reward, residual = self._score(system, support)
                candidates.append((reward, residual, support))
            candidates.sort(reverse=True, key=lambda item: item[0])
            elites = candidates[:elite_count]
            for col, action in enumerate(actions):
                freq = sum(1 for _, _, support in elites if action in support) / elite_count
                probs[col] = 0.25 * probs[col] + 0.75 * min(0.97, max(0.03, freq))
            if elites[0][0] > best_reward:
                best_reward, _, best_support = elites[0]
            history.append({"iteration": iteration, "best_reward": elites[0][0], "best_residual": elites[0][1], "support": [system.names[i] for i in elites[0][2]]})
        beam = {best_support}
        for size in range(1, self.max_terms + 1):
            for combo in itertools.combinations(actions, size):
                beam.add(tuple(sorted(combo)))
        best_support = max(beam, key=lambda s: self._score(system, s)[0])
        coef_scaled = ridge(select_columns(system.theta, best_support), system.y, self.ridge_lambda)
        raw_scales = column_norms(select_columns(system.raw_theta, best_support))
        coef = [c / (s if s else 1.0) for c, s in zip(coef_scaled, raw_scales)]
        cutoff = max(1e-4, 0.01 * max((abs(c) for c in coef), default=0.0))
        pruned = tuple(i for i, c in zip(best_support, coef) if abs(c) > cutoff)
        if pruned:
            coef_scaled = ridge(select_columns(system.theta, pruned), system.y, self.ridge_lambda)
            raw_scales = column_norms(select_columns(system.raw_theta, pruned))
            coef = [c / (s if s else 1.0) for c, s in zip(coef_scaled, raw_scales)]
        else:
            coef = []
        _, residual = self._score(system, pruned)
        return DiscoveryResult(pruned, {system.names[i]: c for i, c in zip(pruned, coef)}, residual, history)
