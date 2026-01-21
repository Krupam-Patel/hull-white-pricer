"""A Monte Carlo and Analytical pricing engine for the HW interest rate model"""
#Fully Done 1-19-2026

import pandas as pd
import numpy as np
from typing import Any

class hwModel:
    def __init__(self, curve: Any, parameters: dict | None = None) -> None:
        self.curve = curve
        defaults = {
            "a": 0.01,
            "sigma": 0.01,
            "r0": curve.inst_fwd_rate(0.0)
        }

        if parameters is None:
            parameters = {}

        self.parameters = {
            "a": parameters.get("a", defaults["a"]),
            "sigma": parameters.get("sigma", defaults["sigma"]),
            "r0": parameters.get("r0", defaults["r0"])
        }

        if self.parameters["a"] <= 0.0:
            raise ValueError(f"Mean reversion 'a' must be positive, got {self.parameters['a']}")
        if self.parameters["sigma"] <= 0.0:
            raise ValueError(f"Volatility 'sigma' must be positive, got {self.parameters['sigma']}")

        self.a: float = self.parameters["a"]
        self.sigma: float = self.parameters["sigma"]
        self.r0: float = self.parameters["r0"]

    def inst_fwd_rate(self, maturity: float) -> float:
        return self.curve.inst_fwd_rate(maturity)

    def discount(self, maturity: float) -> float:
        return self.curve.discount(maturity)

    def fwd_rate(self, start_time: float, end_time: float) -> float:
        return self.curve.fwd_rate(start_time, end_time)

    def theta(self, maturity: float, dt: float = 1e-4) -> float:
        a = self.a
        sigma = self.sigma

        fwd_minus = self.inst_fwd_rate(maturity - dt)
        fwd_plus = self.inst_fwd_rate(maturity + dt)
        df_dt = (fwd_plus - fwd_minus) / (2.0 * dt)

        convexity = (sigma**2 / (2.0 * a**2)) * (1.0 - np.exp(-a * maturity))**2

        return df_dt + a * self.inst_fwd_rate(maturity) + convexity

    def rate_sens(self, start_time: float, end_time: float) -> float:
        a = self.a

        return (1.0 - np.exp(-a * (end_time - start_time))) / a

    def bond_adj_factor(self, start_time: float, end_time: float) -> float:
        a = self.a
        sigma = self.sigma

        risk_sens = self.rate_sens(start_time, end_time)
        df_maturity = self.discount(end_time)
        df_start = self.discount(start_time)

        fwd = self.inst_fwd_rate(start_time)

        adjustment = np.exp(risk_sens * fwd - (sigma**2 / (4.0 * a)) * risk_sens**2 * (1.0 - np.exp(-2.0 * a * start_time)))

        return (df_maturity / df_start) * adjustment


    def short_rate(self, maturity: float, z: float | None = None) -> float:
        if z is None:
            z = np.random.normal()
        r0 = self.r0
        a = self.a
        sigma = self.sigma

        variance = (sigma**2 / (2.0 * a)) * (1.0 - np.exp(-2.0 * a * maturity))
        mean = r0 * np.exp(-a * maturity)

        return mean + np.sqrt(variance) * z

    def short_rate_forward(self, maturity: float, z: float | None = None) -> float:
        if z is None:
            z = np.random.normal()
        a = self.a
        sigma = self.sigma

        variance = (sigma**2 / (2.0 * a)) * (1.0 - np.exp(-2.0 * a * maturity))
        expected_rate = self.inst_fwd_rate(maturity)

        return expected_rate + np.sqrt(variance) * z

class hwSim:
    def __init__(self, model, n_paths: int = 10**5, n_steps: int = 100, seed: int = 2025):
        self.model = model
        self.n_paths = n_paths
        self.n_steps = n_steps
        self.rng = np.random.default_rng(seed)

    def sim_short_rate_direct(self, future_time: float) -> np.ndarray:
        shocks = self.rng.normal(size=self.n_paths)
        simulated_rates = np.array([self.model.short_rate(future_time, z=z_val) for z_val in shocks])

        return simulated_rates

    def sim_short_rate_direct_fwd(self, future_time: float) -> np.ndarray:
        shocks = self.rng.normal(size=self.n_paths)

        sim_rates = np.array([self.model.short_rate_forward(future_time, z=z_val) for z_val in shocks])
        return sim_rates

    def sim_short_rate_euler(self, future_time: float) -> tuple[np.ndarray, np.ndarray]:
        dt = future_time / self.n_steps
        time_grid = np.linspace(0.0, future_time, self.n_steps + 1)
        rate_paths = np.zeros((self.n_paths, self.n_steps + 1))

        r0 = self.model.r0
        a = self.model.a
        sigma = self.model.sigma
        rate_paths[:, 0] = r0

        for time_step in range(1, self.n_steps + 1):
            t_prev = time_grid[time_step - 1]
            shocks = self.rng.normal(size=self.n_paths)
            r_prev = rate_paths[:, time_step - 1]
            drift = (self.model.theta(t_prev) - a * r_prev) * dt
            diffusion = sigma * np.sqrt(dt) * shocks
            rate_paths[:, time_step] = r_prev + drift + diffusion

        return rate_paths, time_grid

    def validate_sim(self, future_time: float) -> pd.DataFrame:
        euler_paths, _ = self.sim_short_rate_euler(future_time)
        euler_final = euler_paths[:, -1]
        direct_paths = self.sim_short_rate_direct(future_time)

        r0 = self.model.r0
        a = self.model.a
        sigma = self.model.sigma

        analytic_mean = (r0 * np.exp(-a * future_time) + self.model.theta(future_time) 
                         - np.exp(-a * future_time) * self.model.theta(0.0))

        analytic_std = np.sqrt((sigma**2 / (2.0 * a)) * (1.0 - np.exp(-2.0 * a * future_time)))

        comparison_data = {"Mean": [np.mean(euler_final), np.mean(direct_paths), analytic_mean],
            "Std Dev": [np.std(euler_final), np.std(direct_paths), analytic_std],}

        df = pd.DataFrame(comparison_data,index=["Euler Simulation", "Direct Simulation", "Analytic"],)

        return df


class hwCurveBuilder:
    def __init__(self, curve, params=None, n_paths: int = 10**5, n_steps: int = 100, seed: int = 2025):
        self.curve = curve
        self.model = hwModel(self.curve, params)
        self.sim = hwSim(self.model, n_paths=n_paths, n_steps=n_steps, seed=seed)

    def short_rate(self, target_maturity: float, fwd_measure: bool = False) -> np.ndarray:
        if fwd_measure:
            return self.sim.sim_short_rate_direct_fwd(target_maturity)
        else:
            return self.sim.sim_short_rate_direct(target_maturity)

    def zero_coupon_bond(self, eval_time: float, maturity: float, fwd_measure: bool = False) -> np.ndarray:
        if fwd_measure:
            rate_at_eval = self.sim.sim_short_rate_direct_fwd(eval_time)
        else:
            rate_at_eval = self.sim.sim_short_rate_direct(eval_time)

        bond_price_prefactor = self.model.bond_adj_factor(eval_time, maturity)
        short_rate_duration = self.model.rate_sens(eval_time, maturity)
        bond_price = bond_price_prefactor * np.exp(-short_rate_duration * rate_at_eval)
        return bond_price

    def disc_factor(self, eval_time: float, maturity: float) -> np.ndarray:
        r_paths, time_grid = self.sim.sim_short_rate_euler(maturity)
        maturity_index = np.searchsorted(time_grid, maturity)
        start_idx = np.searchsorted(time_grid, eval_time)

        dt = time_grid[1] - time_grid[0]
        integral_rates = np.sum(r_paths[:, start_idx:maturity_index] * dt, axis=1)
        df = np.exp(-integral_rates)
        return df

    def inst_fwd_rate(self, start_time: float, end_time: float) -> np.ndarray:
        realized_rate = self.sim.sim_short_rate_direct(start_time)
        market_fwd_end = self.model.inst_fwd_rate(end_time)
        market_fwd_start = self.model.inst_fwd_rate(start_time)

        short_rate_duration = self.model.rate_sens(start_time, end_time)
        a = self.model.a
        sigma = self.model.sigma

        convexity_correction = (sigma**2) * (1.0 - np.exp(-2.0 * a * start_time)) / (2.0 * a)
        predicted_forward = (market_fwd_end + np.exp(-a * (end_time - start_time)) 
                             * (realized_rate - market_fwd_start + convexity_correction * short_rate_duration))
        
        return predicted_forward

    def long_rate(self, start_time: float, end_time: float, fwd_measure: bool = False) -> np.ndarray:
        if fwd_measure:
            spot_short_rate = self.sim.sim_short_rate_direct_fwd(start_time)
        else:
            spot_short_rate = self.sim.sim_short_rate_direct(start_time)

        bond_price_prefactor = self.model.bond_adj_factor(start_time, end_time)
        short_rate_duration = self.model.rate_sens(start_time, end_time)

        maturity_span = end_time - start_time
        yield_intercept = -np.log(bond_price_prefactor) / maturity_span
        yield_slope = short_rate_duration / maturity_span

        yield_to_maturity = yield_intercept + yield_slope * spot_short_rate

        return yield_to_maturity

    def fwd_rate(self, eval_time: float, start_date: float, end_date: float, fwd_measure: bool = False) -> np.ndarray:
        discount_start = self.zero_coupon_bond(eval_time, start_date, fwd_measure=fwd_measure)
        discount_end = self.zero_coupon_bond(eval_time, end_date, fwd_measure=fwd_measure)
        forward_libor = (discount_start / discount_end - 1.0) / (end_date - start_date)
        return forward_libor

    def coupon_bond(self, eval_time: float, coupon_schedule: list[float], coupon_rate: float, principal: float, fwd_measure: bool = False) -> float:
        remaining_coupons = [t for t in coupon_schedule if t >= eval_time]
        if not remaining_coupons:
            return 0.0
        bond_value = 0.0

        for i in range(len(remaining_coupons) - 1):
            accrual_period = remaining_coupons[i + 1] - remaining_coupons[i]
            coupon_cf = coupon_rate * principal * accrual_period
            disc_to_eval = self.zero_coupon_bond(eval_time, remaining_coupons[i], fwd_measure=fwd_measure)
            bond_value += coupon_cf * np.mean(disc_to_eval)

        accrual_last = remaining_coupons[-1] - remaining_coupons[-2] if len(remaining_coupons) > 1 else 0.0
        final_coupon_cf = coupon_rate * principal * accrual_last
        final_disc_to_eval = self.zero_coupon_bond(eval_time, remaining_coupons[-1], fwd_measure=fwd_measure)
        bond_value += (principal + final_coupon_cf) * np.mean(final_disc_to_eval)

        return bond_value


if __name__ == "__main__":
    print("Hull-White model and simulation modules work")
