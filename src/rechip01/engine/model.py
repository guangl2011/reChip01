"""Weibull proportional-hazards reliability model, fit by censored MLE (spec §4).

Survival form per the spec: per-unit scale ``eta = exp(-x·beta)`` (the design
row ``x`` carries an intercept column), shared shape ``k``. Then

    S(t | x) = exp(-(t / eta)^k) = exp(-exp(k (log t + x·beta)))
    P(fail within t | x) = 1 - S(t | x)

so the log-hazard is linear and interpretable: every coefficient reads as the
log-hazard change per standard deviation of the feature. Right-censored units
(units that survived the observation horizon) contribute their survival
probability, never a fabricated event time.

Uncertainty comes from the observed information at the MLE: the covariance of
``(beta, log k)``, computed by central differences of the analytic gradient. A
unit's hazard standard error ``sqrt(x' Cov_beta x)``, shifted ± z·se on the
linear predictor and mapped through the same monotone transform as the point
estimate, yields the calibrated interval — deterministic, no bootstrap noise.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Final

import numpy as np
from numpy.typing import NDArray
from scipy.optimize import minimize
from scipy.stats import norm

# exp(k·z) saturates far below float64 overflow; the ceiling keeps the NLL
# finite at absurd parameter vectors, where the true likelihood is already
# effectively zero and no optimizer step benefits from the unclipped value.
_MAX_EXPONENT: Final[float] = 50.0
_HESSIAN_STEP: Final[float] = 1e-5  # relative step for the central-difference Hessian
_DEFAULT_RIDGE_LAMBDA: Final[float] = 1e-2


def _safe_exp(value: NDArray[np.float64]) -> NDArray[np.float64]:
    """Exponentiate with a ceiling that keeps float64 finite."""
    return np.asarray(np.exp(np.minimum(value, _MAX_EXPONENT)), dtype=np.float64)


def weibull_cloglog(
    linear: NDArray[np.float64], log_time: float, shape: float
) -> NDArray[np.float64]:
    """P(fail within t) from the linear predictor: ``1 - exp(-exp(k(log t + L)))``."""
    kz = np.minimum(shape * (log_time + linear), _MAX_EXPONENT)
    return 1.0 - np.exp(-_safe_exp(kz))


def nll_and_gradient(
    theta: NDArray[np.float64],
    design: NDArray[np.float64],
    log_time: NDArray[np.float64],
    event: NDArray[np.bool_],
    ridge_lambda: float = 0.0,
) -> tuple[float, NDArray[np.float64]]:
    """Negative log-likelihood (plus optional slope ridge) and its analytic gradient.

    ``theta`` is ``(beta, log_k)``; ``design`` rows carry an intercept column;
    ``log_time`` holds log failure times for events and log censoring times
    for censored rows; ``event`` marks observed failures. The ridge penalizes
    slope coefficients only (never the intercept or log-shape): the training
    workload block takes three distinct profile values, so its columns are
    exactly collinear with the intercept and the pure MLE is unidentifiable
    along that direction — a tiny ridge makes the fit strictly convex while
    leaving identifiable directions essentially untouched (shrinkage ~ lambda
    / Fisher information, ~1e-5 at MVP sample sizes).
    """
    beta = theta[:-1]
    k = float(np.exp(theta[-1]))
    linear = design @ beta
    failed = event.astype(np.float64)
    survived = 1.0 - failed

    w = np.minimum(k * (log_time + linear), _MAX_EXPONENT)  # k·z, clipped
    exp_w = _safe_exp(w)

    nll = float(np.sum(failed * (-np.log(k) + log_time - w + exp_w) + survived * exp_w))
    grad_beta = design.T @ (failed * k * (exp_w - 1.0) + survived * k * exp_w)
    if ridge_lambda > 0.0:
        penalty = 0.5 * ridge_lambda * float(beta[1:] @ beta[1:])
        nll += penalty
        grad_beta = grad_beta + np.concatenate([[0.0], ridge_lambda * beta[1:]])
    grad_logk = float(np.sum(failed * (-1.0 + w * (exp_w - 1.0)) + survived * w * exp_w))
    return nll, np.append(grad_beta, grad_logk)


@dataclass(frozen=True)
class WeibullPHModel:
    """Fitted Weibull proportional-hazards model with MLE covariance."""

    coef: NDArray[np.float64]  # (p,) incl. intercept at position 0
    log_shape: float
    covariance: NDArray[np.float64]  # (p+1, p+1) covariance of (beta, log_k)

    @classmethod
    def fit(
        cls,
        design: NDArray[np.float64],
        log_time: NDArray[np.float64],
        event: NDArray[np.bool_],
        ridge_lambda: float = _DEFAULT_RIDGE_LAMBDA,
    ) -> WeibullPHModel:
        """Fit by penalized maximum likelihood (slope-only ridge).

        ``design`` must already carry an intercept column; ``log_time`` holds
        log failure times for events and log censoring times for censored rows.
        """
        if int(event.sum()) == 0:
            raise ValueError("no observed failures in training data; model is unfittable")

        theta0 = np.zeros(design.shape[1] + 1)
        theta0[-1] = np.log(1.5)  # Weibull-shape starting guess
        result = minimize(
            nll_and_gradient,
            theta0,
            args=(design, log_time, event, ridge_lambda),
            method="L-BFGS-B",
            jac=True,
            options={"maxiter": 2000, "ftol": 1e-12, "gtol": 1e-8},
        )
        if not result.success or not np.all(np.isfinite(result.x)):
            raise RuntimeError(f"Weibull PH fit did not converge: {result.message}")

        covariance = cls._covariance_from_hessian(result.x, design, log_time, event, ridge_lambda)
        return cls(
            coef=np.asarray(result.x[:-1], dtype=np.float64),
            log_shape=float(result.x[-1]),
            covariance=covariance,
        )

    @staticmethod
    def _covariance_from_hessian(
        theta: NDArray[np.float64],
        design: NDArray[np.float64],
        log_time: NDArray[np.float64],
        event: NDArray[np.bool_],
        ridge_lambda: float = _DEFAULT_RIDGE_LAMBDA,
    ) -> NDArray[np.float64]:
        """Inverse observed information via a central-difference Hessian."""
        n_params = theta.shape[0]
        hessian = np.zeros((n_params, n_params), dtype=np.float64)
        steps = np.maximum(np.abs(theta), 1.0) * _HESSIAN_STEP
        for column in range(n_params):
            up, down = theta.copy(), theta.copy()
            up[column] += steps[column]
            down[column] -= steps[column]
            grad_up = nll_and_gradient(up, design, log_time, event, ridge_lambda)[1]
            grad_down = nll_and_gradient(down, design, log_time, event, ridge_lambda)[1]
            hessian[:, column] = (np.asarray(grad_up) - np.asarray(grad_down)) / (
                2.0 * steps[column]
            )
        hessian = (hessian + hessian.T) / 2.0  # enforce symmetry
        # A non-positive-definite Hessian (unidentifiable optimum) must fail
        # loudly: inverting it yields negative/huge variances that silently
        # destroy every downstream interval.
        try:
            np.linalg.cholesky(hessian)
        except np.linalg.LinAlgError as error:
            raise RuntimeError(
                "Weibull PH Hessian is not positive definite at the optimum; "
                "parameters are not identifiable from this training data"
            ) from error
        try:
            covariance = np.linalg.inv(hessian)
        except np.linalg.LinAlgError as error:
            raise RuntimeError(
                "Weibull PH Hessian is singular at the optimum; parameters are not "
                "identifiable from this training data"
            ) from error
        if not np.all(np.isfinite(covariance)) or np.any(np.diag(covariance) <= 0):
            raise RuntimeError("Weibull PH covariance is invalid at the optimum")
        return np.asarray(covariance, dtype=np.float64)

    @property
    def shape(self) -> float:
        """Weibull shape parameter k."""
        return float(np.exp(self.log_shape))

    def linear_predictor(self, design_row: NDArray[np.float64]) -> NDArray[np.float64]:
        """x·beta for one row or a matrix of rows."""
        return np.asarray(design_row @ self.coef, dtype=np.float64)

    def linear_predictor_se(self, design_row: NDArray[np.float64]) -> float:
        """Standard error of x·beta from the beta block of the MLE covariance.

        Shape uncertainty is absorbed in the point estimates; intervals
        quantify hazard-parameter uncertainty, the dominant term for per-unit
        predictions (documented simplification of the delta method).
        """
        p = self.coef.shape[0]
        beta_cov = self.covariance[:p, :p]
        variance = float(design_row @ beta_cov @ design_row)
        return float(np.sqrt(max(variance, 0.0)))

    def p_fail(self, design: NDArray[np.float64], horizon_days: int) -> NDArray[np.float64]:
        """P(fail within ``horizon_days``) for one design row or a matrix of rows."""
        linear = np.atleast_1d(self.linear_predictor(design))
        return weibull_cloglog(linear, float(np.log(horizon_days)), self.shape)

    def p_fail_at_linear(self, linear: float, horizon_days: int) -> float:
        """P(fail) for a raw linear-predictor value (interval arithmetic helper)."""
        log_time = float(np.log(horizon_days))
        return float(weibull_cloglog(np.asarray([linear]), log_time, self.shape)[0])

    def interval(
        self,
        design_row: NDArray[np.float64],
        horizon_days: int,
        ci_level: float,
        se_widening: float,
    ) -> tuple[float, float, float]:
        """Calibrated interval ``(low, p_fail, high)`` for one design row.

        The linear predictor's standard error is inflated by ``se_widening``
        (the missingness/OOD widening enters here), shifted ± z·se on the
        linear predictor, and mapped through the same monotone transform as
        the point estimate — P(fail) is increasing in the linear predictor, so
        the endpoints order correctly; the final clamp is rounding insurance.
        """
        if not 0.0 < ci_level < 1.0:
            raise ValueError(f"ci_level must be in (0, 1), got {ci_level}")
        if se_widening < 1.0:
            raise ValueError(f"se_widening must be >= 1.0, got {se_widening}")

        linear = float(self.linear_predictor(design_row))
        se = self.linear_predictor_se(design_row) * se_widening
        z_quantile = float(norm.ppf(0.5 + ci_level / 2.0))
        p_fail = self.p_fail_at_linear(linear, horizon_days)
        p_low = self.p_fail_at_linear(linear - z_quantile * se, horizon_days)
        p_high = self.p_fail_at_linear(linear + z_quantile * se, horizon_days)
        return min(p_low, p_fail), p_fail, max(p_high, p_fail)


__all__ = ["WeibullPHModel", "nll_and_gradient", "weibull_cloglog"]
