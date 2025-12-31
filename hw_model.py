'''A Monte Carlo and Analytical pricing engine for the HW interest rate model'''

import pandas as pd
import numpy as np
class hwModel:
    def __init__(self, curve, parameters=None):
        self.curve = curve
        defaults = {'a': 0.01, 'sigma': 0.01, 'r0': curve.inst_fwd_rate(0)}
        if parameters is None:
            parameters = {}
        self.parameters = {
            'a': parameters.get('a', defaults['a']),
            'sigma': parameters.get('sigma', defaults['sigma']),
            'r0': parameters.get('r0', defaults['r0'])
        }
        
        if self.parameters['a'] <= 0:
            raise ValueError(f"Mean reversion 'a' must be positive, got {self.parameters['a']}")
        if self.parameters['sigma'] <= 0:
            raise ValueError(f"Volatility 'sigma' must be positive, got {self.parameters['sigma']}")

    def inst_fwd_rate(self, maturity):
        return self.curve.inst_fwd_rate(maturity)

    def discount(self, maturity):
        return self.curve.discount(maturity)
    
    def fwd_rate(self, start_time, end_time):
        return self.curve.forward_rate(start_time, end_time)

    def theta(self, maturity, dt=1e-6):
        a = self.parameters['a']
        sigma = self.parameters['sigma']
        
        fwd_rate_now = self.inst_fwd_rate(maturity)
        fwd_rate_next = self.inst_fwd_rate(maturity + dt)
        
        dt_fwd = (fwd_rate_next - fwd_rate_now) / dt
        convexity = (sigma**2 / (2 * a**2)) * (1 - np.exp(-a * maturity))**2
        
        return dt_fwd + a * fwd_rate_now + convexity

    def rate_sens(self, start_time, end_time):
        a = self.parameters['a']
        return (1 - np.exp(-a * (end_time - start_time))) / a

    def bond_adj_factor(self, start_time, end_time):
        a = self.parameters['a']
        sigma = self.parameters['sigma']
        
        risk_sens = self.rate_sens(start_time, end_time)
        df_maturity = self.discount(end_time)
        df_start = self.discount(start_time)
        
        fwd = self.inst_fwd_rate(start_time)
        
        adjustment = np.exp(
            risk_sens * fwd - (sigma**2 / (4*a)) * risk_sens**2 * (1 - np.exp(-2*a*start_time))
        )
        
        return (df_maturity / df_start) * adjustment

    def short_rate(self, maturity, z=None):
        if z is None:
            z = np.random.normal()

        r0 = self.parameters['r0']
        a = self.parameters['a']
        sigma = self.parameters['sigma']
        variance = (sigma**2 / (2 * a)) * (1 - np.exp(-2 * a * maturity))

        mean_rate = r0 * np.exp(-a * maturity) + self.theta(maturity) - np.exp(-a * maturity) * self.theta(0)

        return mean_rate + np.sqrt(variance) * z

    def short_rate_forward(self, maturity, z=None):
        if z is None:
            z = np.random.normal()

        a = self.parameters['a']
        sigma = self.parameters['sigma']
        variance = (sigma**2 / (2 * a)) * (1 - np.exp(-2 * a * maturity))
        expected_rate = self.curve.inst_fwd_rate(maturity)
        return expected_rate + np.sqrt(variance) * z


class hwSim:
    def __init__(self, model, n_paths=10**5, n_steps=100, seed=2025):
        self.model = model
        self.n_paths = n_paths
        self.n_steps = n_steps
        np.random.seed(seed)

    def sim_short_rate_direct(self, future_time):
        shocks = np.random.normal(size=self.n_paths)
        simulated_rates = np.array([self.model.short_rate(future_time, z=z_val) for z_val in shocks])
        return simulated_rates

    def sim_short_rate_direct_fwd(self, future_time):
        shocks = np.random.normal(size=self.n_paths)
        sim_rates = np.array([self.model.short_rate_fwd(future_time, z=z_val) for z_val in shocks])
        return sim_rates

    def sim_short_rate_euler(self, future_time):
        dt = future_time / self.n_steps
        time_grid = np.linspace(0, future_time, self.n_steps + 1)
        rate_deviations = np.zeros((self.n_paths, self.n_steps + 1))
        rate_paths = np.zeros_like(rate_deviations)

        rate_deviations[:, 0] = self.model.parameters['r0'] - self.model.theta(0)
        rate_paths[:, 0] = self.model.parameters['r0']

        for time_step in range(1, self.n_steps + 1):
            shocks = np.random.normal(size=self.n_paths)
            prev_dev = rate_deviations[:, time_step - 1]
            mean_reversion_pull = self.model.parameters['a'] * prev_dev * dt
            vol_shock = self.model.parameters['sigma'] * np.sqrt(dt) * shocks
            rate_deviations[:, time_step] = prev_dev - mean_reversion_pull + vol_shock
            theta_val = self.model.theta(time_grid[time_step])
            rate_paths[:, time_step] = rate_deviations[:, time_step] + theta_val

        return rate_paths, time_grid

    def validate_sim(self, future_time):
        euler_paths, _ = self.sim.sim_short_rate_euler(future_time)
        euler_final = euler_paths[:, -1]
        direct_paths = self.sim.sim_short_rate_direct(future_time)

        analytic_mean = (self.model.parameters['r0'] * np.exp(-self.model.parameters['a'] * future_time) 
                         + self.model.theta(future_time) - np.exp(-self.model.parameters['a'] 
                                                                  * future_time) * self.model.theta(0))
        analytic_std = np.sqrt((self.model.parameters['sigma']**2) / (2 * self.model.parameters['a']) 
                              * (1 - np.exp(-2 * self.model.parameters['a'] * future_time)))

        comparison_data = {
            "Mean": [np.mean(euler_final), np.mean(direct_paths), analytic_mean],
            "Std Dev": [np.std(euler_final), np.std(direct_paths), analytic_std]
        }
        df = pd.DataFrame(comparison_data, index=["Euler Simulation", "Direct Simulation", "Analytic"])
        return df

class hwCurveBuilder:
    def __init__(self, curve, params=None, n_paths=10**5, n_steps=100, seed=2025):
        self.curve = curve
        self.model = hwModel(self.curve, params)
        self.sim = hwSim(self.model, n_paths=n_paths, n_steps=n_steps, seed=seed)

    def short_rate(self, target_maturity, fwd_measure=False):
        if fwd_measure:
            return self.sim.sim_short_rate_direct_fwd(target_maturity)
        else:
            return self.sim.sim_short_rate_direct(target_maturity)

    def zero_coupon_bond(self, eval_time, maturity, fwd_measure=False):
        if fwd_measure:
            rate_at_eval = self.sim.sim_short_rate_direct_fwd(eval_time)
        else:
            rate_at_eval = self.sim.sim_short_rate_direct(eval_time)
        
        A_factor = self.model.A(eval_time, maturity)
        B_factor = self.model.B(eval_time, maturity)
        bond_price = A_factor * np.exp(-B_factor * rate_at_eval)
        return bond_price

    def disc_factor(self, eval_time, maturity):
        r_paths, time_grid = self.sim.sim_short_rate_euler(maturity)
        maturity_index = np.searchsorted(time_grid, maturity)
        start_idx = np.searchsorted(time_grid, eval_time) = np.searchsorted(time_grid, eval_time)

        dt = time_grid[1] - time_grid[0]
        integral_rates = np.sum(r_paths[:, start_idx:maturity_index] * dt, axis=1)
        df = np.exp(-integral_rates)
        return df

    def inst_fwd_rate(self, start_time, end_time):
        realized_rate = self.sim.sim_short_rate_direct(start_time)
        market_fwd_end = self.model.inst_fwd_rate(end_time)
        market_fwd_start = self.model.inst_fwd_rate(start_time)
        B_factor = self.model.B(start_time, end_time)
        a = self.model.parameters['a']
        sigma = self.model.parameters['sigma']
        convexity_adj = (sigma**2) * (1 - np.exp(-2 * a * start_time)) / (2 * a)
        predicted_fwd = (market_fwd_end + np.exp(-a * (end_time - start_time))
            * (realized_rate - market_fwd_start + convexity_adj * B_factor)
        )
        return predicted_fwd

    def long_rate(self, start_time, end_time, fwd_measure=False):
        if fwd_measure:
            spot_rate = self.sim.sim_short_rate_direct_fwd(start_time)
        else:
            spot_rate = self.sim.sim_short_rate_direct(start_time)

        
        # FIX THIS PART STILL 

        
        A_factor = self.model.A(start_time, end_time)
        B_factor = self.model.B(start_time, end_time)
        alpha = -np.log(A_factor) / (end_time - start_time)
        beta = B_factor / (end_time - start_time)
        ytm = alpha + beta * spot_rate
        return ytm
    
    def fwd_rate(self, eval_time, start_date, end_date, fwd_measure=False):
        bond_at_start = self.zero_coupon_bond(eval_time, start_date, fwd_measure=fwd_measure)
        bond_at_end = self.zero_coupon_bond(eval_time, end_date, fwd_measure=fwd_measure)
        libor_rate = (1.0 / (end_date - start_date)) * (bond_at_start / bond_at_end - 1.0)
        return libor_rate

    def coupon_bond(self, eval_time, coupon_schedule, coupon_rate, principal, fwd_measure=False):
        valid_coupons = [t for t in coupon_schedule if t >= eval_time]
        if not valid_coupons:
            return 0.0
        
        bond_value = 0.0
        
        for i in range(len(valid_coupons) - 1):
            accrual_period = valid_coupons[i+1] - valid_coupons[i]
            coupon_amt = coupon_rate * principal * accrual_period
            df_coupon = self.zero_coupon_bond(eval_time, valid_coupons[i], fwd_measure=fwd_measure)
            bond_value += coupon_amt * np.mean(df_coupon)
        
        accrual_last = valid_coupons[-1] - valid_coupons[-2] if len(valid_coupons) > 1 else 0.0
        final_coupon = coupon_rate * principal * accrual_last
        df_final = self.zero_coupon_bond(eval_time, valid_coupons[-1], fwd_measure=fwd_measure)
        bond_value += (principal + final_coupon) * np.mean(df_final)
        
        return bond_value


if __name__ == "__main__":
    print("Hull-White model and simulation modules work")

