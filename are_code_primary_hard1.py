"""Genetic optimization of linear-model weights for array inputs.

This module stays self-contained on purpose: it only requires NumPy, so the
example can be executed and validated without depending on scikit-learn.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Tuple

import numpy as np


ArrayLike = np.ndarray


@dataclass
class TrainingMetrics:
    """Summary of the best generation found during training."""

    generation: int
    train_mse: float
    validation_mse: float


@dataclass
class GeneticLinearRegressor:
    """Optimize linear weights with a genetic algorithm.

    The model standardizes features and targets internally to make the search
    space much more stable. After training, it converts the best solution back
    to the original scale so predictions stay intuitive.
    """

    population_size: int = 120
    generations: int = 160
    mutation_rate: float = 0.12
    mutation_scale: float = 0.08
    crossover_rate: float = 0.9
    elite_fraction: float = 0.1
    tournament_size: int = 4
    validation_split: float = 0.2
    early_stopping_rounds: int = 25
    random_state: int | None = 42
    coefficients_: ArrayLike | None = field(default=None, init=False)
    intercept_: float | None = field(default=None, init=False)
    history_: List[TrainingMetrics] = field(default_factory=list, init=False)

    def fit(self, X: ArrayLike, y: ArrayLike) -> "GeneticLinearRegressor":
        """Fit the model on 2D feature arrays and a 1D target array."""

        X, y = self._validate_inputs(X, y)
        rng = np.random.default_rng(self.random_state)
        X_train, X_val, y_train, y_val = self._train_validation_split(X, y, rng)

        (
            X_train_std,
            y_train_std,
            X_mean,
            X_scale,
            y_mean,
            y_scale,
        ) = self._standardize_training_data(X_train, y_train)
        X_val_std = self._standardize_features(X_val, X_mean, X_scale)
        y_val_std = self._standardize_targets(y_val, y_mean, y_scale)

        X_train_aug = self._augment_bias(X_train_std)
        X_val_aug = self._augment_bias(X_val_std)

        population = rng.normal(
            loc=0.0,
            scale=0.5,
            size=(self.population_size, X_train_aug.shape[1]),
        )

        elite_count = max(2, int(self.population_size * self.elite_fraction))
        best_solution = population[0].copy()
        best_val_mse = float("inf")
        rounds_without_improvement = 0
        self.history_ = []

        for generation in range(1, self.generations + 1):
            train_mse = self._population_mse(population, X_train_aug, y_train_std)
            val_mse = self._population_mse(population, X_val_aug, y_val_std)
            ranked_indices = np.argsort(val_mse)
            population = population[ranked_indices]
            train_mse = train_mse[ranked_indices]
            val_mse = val_mse[ranked_indices]

            current_best_val = float(val_mse[0])
            self.history_.append(
                TrainingMetrics(
                    generation=generation,
                    train_mse=float(train_mse[0]),
                    validation_mse=current_best_val,
                )
            )

            if current_best_val < best_val_mse:
                best_val_mse = current_best_val
                best_solution = population[0].copy()
                rounds_without_improvement = 0
            else:
                rounds_without_improvement += 1

            if rounds_without_improvement >= self.early_stopping_rounds:
                break

            next_population = [candidate.copy() for candidate in population[:elite_count]]
            while len(next_population) < self.population_size:
                parent_a = self._tournament_select(population, val_mse, rng)
                parent_b = self._tournament_select(population, val_mse, rng)
                child_a, child_b = self._crossover(parent_a, parent_b, rng)
                child_a = self._mutate(child_a, rng)
                child_b = self._mutate(child_b, rng)
                next_population.append(child_a)
                if len(next_population) < self.population_size:
                    next_population.append(child_b)

            population = np.asarray(next_population, dtype=float)

        self.coefficients_, self.intercept_ = self._destandardize_solution(
            best_solution,
            X_mean,
            X_scale,
            y_mean,
            y_scale,
        )
        return self

    def predict(self, X: ArrayLike) -> ArrayLike:
        """Predict targets for a 2D feature array."""

        if self.coefficients_ is None or self.intercept_ is None:
            raise RuntimeError("The model must be fitted before calling predict().")

        X = np.asarray(X, dtype=float)
        if X.ndim == 1:
            X = X.reshape(1, -1)
        if X.ndim != 2:
            raise ValueError("X must be a 2D array.")
        if X.shape[1] != self.coefficients_.shape[0]:
            raise ValueError("X has a different number of features than the fitted model.")

        return X @ self.coefficients_ + self.intercept_

    def score(self, X: ArrayLike, y: ArrayLike) -> Dict[str, float]:
        """Return regression metrics for the provided data."""

        y = np.asarray(y, dtype=float)
        predictions = self.predict(X)
        mse = float(np.mean((predictions - y) ** 2))
        variance = float(np.var(y))
        r2 = 1.0 if variance == 0 else float(1.0 - mse / variance)
        return {"mse": mse, "r2": r2}

    def best_generation(self) -> TrainingMetrics:
        """Return the best stored generation record."""

        if not self.history_:
            raise RuntimeError("No training history is available before fit().")
        return min(self.history_, key=lambda item: item.validation_mse)

    def _validate_inputs(self, X: ArrayLike, y: ArrayLike) -> Tuple[ArrayLike, ArrayLike]:
        X = np.asarray(X, dtype=float)
        y = np.asarray(y, dtype=float).reshape(-1)

        if X.ndim != 2:
            raise ValueError("X must be a 2D array of shape (n_samples, n_features).")
        if y.ndim != 1:
            raise ValueError("y must be a 1D target array.")
        if len(X) != len(y):
            raise ValueError("X and y must contain the same number of samples.")
        if len(X) < 10:
            raise ValueError("At least 10 samples are required for robust optimization.")
        if not np.isfinite(X).all() or not np.isfinite(y).all():
            raise ValueError("X and y must only contain finite numeric values.")
        return X, y

    def _train_validation_split(
        self,
        X: ArrayLike,
        y: ArrayLike,
        rng: np.random.Generator,
    ) -> Tuple[ArrayLike, ArrayLike, ArrayLike, ArrayLike]:
        n_samples = len(X)
        indices = rng.permutation(n_samples)
        val_count = max(1, int(n_samples * self.validation_split))
        train_indices = indices[val_count:]
        val_indices = indices[:val_count]
        return X[train_indices], X[val_indices], y[train_indices], y[val_indices]

    def _standardize_training_data(
        self,
        X: ArrayLike,
        y: ArrayLike,
    ) -> Tuple[ArrayLike, ArrayLike, ArrayLike, ArrayLike, float, float]:
        X_mean = X.mean(axis=0)
        X_scale = np.where(X.std(axis=0) == 0, 1.0, X.std(axis=0))
        y_mean = float(y.mean())
        y_scale = float(y.std()) or 1.0

        X_std = self._standardize_features(X, X_mean, X_scale)
        y_std = self._standardize_targets(y, y_mean, y_scale)
        return X_std, y_std, X_mean, X_scale, y_mean, y_scale

    @staticmethod
    def _standardize_features(X: ArrayLike, mean: ArrayLike, scale: ArrayLike) -> ArrayLike:
        return (X - mean) / scale

    @staticmethod
    def _standardize_targets(y: ArrayLike, mean: float, scale: float) -> ArrayLike:
        return (y - mean) / scale

    @staticmethod
    def _augment_bias(X: ArrayLike) -> ArrayLike:
        bias = np.ones((X.shape[0], 1), dtype=float)
        return np.hstack((X, bias))

    @staticmethod
    def _population_mse(population: ArrayLike, X: ArrayLike, y: ArrayLike) -> ArrayLike:
        predictions = X @ population.T
        residuals = predictions - y[:, None]
        return np.mean(residuals**2, axis=0)

    def _tournament_select(
        self,
        population: ArrayLike,
        losses: ArrayLike,
        rng: np.random.Generator,
    ) -> ArrayLike:
        candidate_indices = rng.choice(
            len(population),
            size=min(self.tournament_size, len(population)),
            replace=False,
        )
        winner_index = candidate_indices[np.argmin(losses[candidate_indices])]
        return population[winner_index].copy()

    def _crossover(
        self,
        parent_a: ArrayLike,
        parent_b: ArrayLike,
        rng: np.random.Generator,
    ) -> Tuple[ArrayLike, ArrayLike]:
        if rng.random() >= self.crossover_rate:
            return parent_a.copy(), parent_b.copy()

        alpha = rng.random(parent_a.shape[0])
        child_a = alpha * parent_a + (1.0 - alpha) * parent_b
        child_b = alpha * parent_b + (1.0 - alpha) * parent_a
        return child_a, child_b

    def _mutate(self, candidate: ArrayLike, rng: np.random.Generator) -> ArrayLike:
        mutation_mask = rng.random(candidate.shape[0]) < self.mutation_rate
        if not mutation_mask.any():
            return candidate
        candidate = candidate.copy()
        candidate[mutation_mask] += rng.normal(
            loc=0.0,
            scale=self.mutation_scale,
            size=int(mutation_mask.sum()),
        )
        return candidate

    @staticmethod
    def _destandardize_solution(
        solution: ArrayLike,
        X_mean: ArrayLike,
        X_scale: ArrayLike,
        y_mean: float,
        y_scale: float,
    ) -> Tuple[ArrayLike, float]:
        std_weights = solution[:-1]
        std_bias = float(solution[-1])

        coefficients = y_scale * std_weights / X_scale
        intercept = y_mean + y_scale * std_bias - float(np.dot(coefficients, X_mean))
        return coefficients, intercept


def make_synthetic_regression(
    n_samples: int = 600,
    n_features: int = 8,
    noise: float = 0.15,
    random_state: int = 42,
) -> Tuple[ArrayLike, ArrayLike, ArrayLike]:
    """Create a stable synthetic regression problem using NumPy only."""

    rng = np.random.default_rng(random_state)
    X = rng.normal(size=(n_samples, n_features))
    true_weights = rng.normal(loc=0.0, scale=2.0, size=n_features)
    bias = rng.normal(loc=0.0, scale=0.5)
    y = X @ true_weights + bias + rng.normal(loc=0.0, scale=noise, size=n_samples)
    return X, y, true_weights


def train_test_split_numpy(
    X: ArrayLike,
    y: ArrayLike,
    test_size: float = 0.2,
    random_state: int = 42,
) -> Tuple[ArrayLike, ArrayLike, ArrayLike, ArrayLike]:
    """Simple NumPy-only train/test split helper."""

    rng = np.random.default_rng(random_state)
    indices = rng.permutation(len(X))
    test_count = max(1, int(len(X) * test_size))
    test_indices = indices[:test_count]
    train_indices = indices[test_count:]
    return X[train_indices], X[test_indices], y[train_indices], y[test_indices]


def main() -> None:
    X, y, true_weights = make_synthetic_regression(
        n_samples=800,
        n_features=10,
        noise=0.2,
        random_state=7,
    )
    X_train, X_test, y_train, y_test = train_test_split_numpy(
        X,
        y,
        test_size=0.2,
        random_state=7,
    )

    model = GeneticLinearRegressor(
        population_size=140,
        generations=180,
        mutation_rate=0.14,
        mutation_scale=0.07,
        early_stopping_rounds=30,
        random_state=7,
    )
    model.fit(X_train, y_train)

    metrics = model.score(X_test, y_test)
    best = model.best_generation()
    coefficient_alignment = float(
        np.corrcoef(model.coefficients_, true_weights)[0, 1]
    )

    print("Best generation:", best.generation)
    print("Validation MSE:", round(best.validation_mse, 6))
    print("Test MSE:", round(metrics["mse"], 6))
    print("Test R2:", round(metrics["r2"], 6))
    print("Coefficient alignment:", round(coefficient_alignment, 6))


if __name__ == "__main__":
    main()
