"""Tests for the generated genetic weight optimizer module."""

import unittest

import numpy as np

from are_code_primary_hard import (
    GeneticLinearRegressor,
    make_synthetic_regression,
    train_test_split_numpy,
)


class GeneticLinearRegressorTests(unittest.TestCase):
    def test_fit_beats_zero_baseline_on_synthetic_regression(self):
        X, y, true_weights = make_synthetic_regression(
            n_samples=500,
            n_features=6,
            noise=0.12,
            random_state=11,
        )
        X_train, X_test, y_train, y_test = train_test_split_numpy(
            X,
            y,
            test_size=0.25,
            random_state=11,
        )

        model = GeneticLinearRegressor(
            population_size=100,
            generations=140,
            mutation_rate=0.12,
            mutation_scale=0.06,
            early_stopping_rounds=20,
            random_state=11,
        )
        model.fit(X_train, y_train)

        predictions = model.predict(X_test)
        mse = float(np.mean((predictions - y_test) ** 2))
        baseline_mse = float(np.mean((np.zeros_like(y_test) - y_test) ** 2))
        alignment = float(np.corrcoef(model.coefficients_, true_weights)[0, 1])

        self.assertLess(mse, baseline_mse * 0.15)
        self.assertGreater(alignment, 0.95)

    def test_predict_requires_fit(self):
        model = GeneticLinearRegressor()
        with self.assertRaises(RuntimeError):
            model.predict(np.ones((3, 2), dtype=float))

    def test_invalid_inputs_are_rejected(self):
        model = GeneticLinearRegressor()
        with self.assertRaises(ValueError):
            model.fit(np.array([1.0, 2.0, 3.0]), np.array([1.0, 2.0, 3.0]))

        with self.assertRaises(ValueError):
            model.fit(
                np.ones((12, 3), dtype=float),
                np.ones((10,), dtype=float),
            )


if __name__ == "__main__":
    unittest.main()
