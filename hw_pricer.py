"""Interest Rate Derivatives Pricing"""
# Still working on

import numpy as np
from scipy.stats import norm
from scipy.optimize import brentq
from hw_model import hwCurveBuilder


class hwPricer:
    def __init__(self, curve, n_paths: int = 10**5, n_steps: int = 252, seed: int = 2025, hw_params: dict | None = None):
        self.curve = curve
        self.curve_sim = hwCurveBuilder(curve, params=hw_params, n_paths=n_paths, n_steps=n_steps, seed=seed)
        self.model = self.curve_sim.model

    def set_sim(self, n_paths=None, n_steps=None, seed=None):
        if n_paths is not None:
            if not isinstance(n_paths, (int, float)) or n_paths <= 0:
                raise ValueError(f"n_paths got to be positive, got {n_paths}")
            self.curve_sim.sim.n_paths = int(n_paths)

        if n_steps is not None:
            if not isinstance(n_steps, (int, float)) or n_steps <= 0:
                raise ValueError(f"n_steps got to be positive, got {n_steps}")
            self.curve_sim.sim.n_steps = int(n_steps)

        if seed is not None:
            if not isinstance(seed, (int, float)):
                raise ValueError(f"seed got to be an integer, got {seed}")
            np.random.seed(int(seed))

    def validate_times(self, option_expiry: float, bond_maturity: float) -> None:
        if not isinstance(option_expiry, (int, float)) or option_expiry < 0:
            raise ValueError(f"option_expiry must be non-negative float, got {option_expiry}")
        if not isinstance(bond_maturity, (int, float)) or bond_maturity < 0:
            raise ValueError(f"bond_maturity must be non-negative float, got {bond_maturity}")
        if bond_maturity < option_expiry:
            raise ValueError(f"bond_maturity ({bond_maturity}) must be >= option_expiry ({option_expiry})")

    def validate_positive(self, value: float, name: str) -> None:
        if not isinstance(value, (int, float)) or value <= 0:
            raise ValueError(f"{name} must be positive, got {value}")

    def price_zero_bond_put_mc(self, option_expiry: float, bond_maturity: float, strike: float) -> float:
        df = self.model.discount(option_expiry)
        bond_price_sim = self.curve_sim.zero_coupon_bond(option_expiry, bond_maturity, fwd_measure=True)
        payoff = np.maximum(strike - bond_price_sim, 0)

        return np.mean(df * payoff)

    def price_zero_bond_put_analytical(self, option_expiry: float, bond_maturity: float, strike: float) -> float:
        sigma = self.model.parameters['sigma']
        a = self.model.parameters['a']
        bond_sens = self.model.rate_sens(option_expiry, bond_maturity)
        bond_price_maturity = self.model.discount(bond_maturity)
        bond_price_expiry = self.model.discount(option_expiry)
        vol_bond = sigma * np.sqrt((1 - np.exp(-2 * a * option_expiry)) / (2 * a)) * bond_sens
        h = (1 / vol_bond) * np.log(bond_price_maturity / (strike * bond_price_expiry)) + 0.5 * vol_bond
        return strike * bond_price_expiry * norm.cdf(-h + vol_bond) - bond_price_maturity * norm.cdf(-h)

    def zero_bond_put(self, option_expiry: float, bond_maturity: float, strike: float, mc: bool = False) -> float:
        self._validate_times(option_expiry, bond_maturity)
        self._validate_positive(strike, "strike")
        if option_expiry == 0.0:
            bond_price = self.model.discount(bond_maturity)
            return max(strike - bond_price, 0.0)
        if mc: # This is monte carlo btw
            return self._price_zero_bond_put_mc(option_expiry, bond_maturity, strike)
        return self._price_zero_bond_put_analytical(option_expiry, bond_maturity, strike)
# Working here 
    def price_zero_bond_call_mc(self, option_expiry: float, bond_maturity: float, strike: float) -> float:
        df = self.model.discount(option_expiry)
        bond_price_sim = self.curve_sim.zero_coupon_bond(option_expiry, bond_maturity, fwd_measure=True)
        payoff = np.maximum(bond_price_sim - strike, 0)
        return np.mean(df * payoff)

    def price_zero_bond_call_analytical(self, option_expiry: float, bond_maturity: float, strike: float) -> float:
        sigma = self.model.parameters['sigma']
        a = self.model.parameters['a']
        bond_sens = self.model.rate_sens(option_expiry, bond_maturity)
        bond_price_maturity = self.model.discount(bond_maturity)
        bond_price_expiry = self.model.discount(option_expiry)
        vol_bond = sigma * np.sqrt((1 - np.exp(-2 * a * option_expiry)) / (2 * a)) * bond_sens
        h = (1 / vol_bond) * np.log(bond_price_maturity / (strike * bond_price_expiry)) + 0.5 * vol_bond
        return bond_price_maturity * norm.cdf(h) - strike * bond_price_expiry * norm.cdf(h - vol_bond)

    def zero_bond_call(self, option_expiry: float, bond_maturity: float, strike: float, mc: bool = False) -> float:
        self.validate_times(option_expiry, bond_maturity)
        self.validate_positive(strike, "strike")
        
        if mc:
            return self._price_zero_bond_call_mc(option_expiry, bond_maturity, strike)
        else:
            return self.price_zero_bond_call_analytical(option_expiry, bond_maturity, strike)

    def caplet(self, fixing_time: float, payment_time: float, notional: float, strike: float, method: str = 'js') -> float:
        self.validate_times(fixing_time, payment_time)
        self.validate_positive(notional, "notional")
        self.validate_positive(strike, "strike")
        
        accrual_period = payment_time - fixing_time
        k_bond = 1 + strike * accrual_period

        if method == 'mc':
            forward_rate = self.curve_sim.inst_fwd_rate(fixing_time, fixing_time, payment_time)
            payoff = accrual_period * np.maximum(forward_rate - strike, 0)
            discount_payment = self.model.discount(payment_time)
            caplet_value = discount_payment * np.mean(payoff)

        elif method == 'js':
            put_price = self.zero_bond_put(fixing_time, payment_time, 1 / k_bond, mc=False)
            caplet_value = k_bond * put_price

        elif method == 'cf':
            sigma = self.model.parameters['sigma']
            mean_reversion = self.model.parameters['a']
            bond_sensitivity = self.model.rate_sens(fixing_time, payment_time)
            discount_payment = self.model.discount(payment_time)
            discount_fixing = self.model.discount(fixing_time)
            vol_bond = sigma * np.sqrt((1 - np.exp(-2 * mean_reversion * fixing_time)) / (2 * mean_reversion)) * bond_sensitivity
            h = (1 / vol_bond) * np.log(discount_payment * k_bond / discount_fixing) + 0.5 * vol_bond
            caplet_value = discount_fixing * norm.cdf(-h + vol_bond) - k_bond * discount_payment * norm.cdf(-h)
        else:
            raise ValueError(f"method must be 'mc', 'js', or 'cf', got {method}")

        return notional * caplet_value

    def cap(self, payment_schedule: np.ndarray, notional: float, strike: float, mc: bool = False) -> float:
        self.validate_positive(notional, "notional")
        self.validate_positive(strike, "strike")
        
        cap_value = 0.0
        if mc:
            for i in range(1, len(payment_schedule)):
                fixing_time = payment_schedule[i - 1]
                payment_time = payment_schedule[i]
                accrual_period = payment_time - fixing_time
                forward_rate = self.curve_sim.fwd_rate(fixing_time, fixing_time, payment_time, fwd_measure=True)
                payoff = accrual_period * np.maximum(forward_rate - strike, 0)
                discount_payment = self.model.discount(payment_time)
                cap_value += discount_payment * np.mean(payoff)
        else:
            for i in range(1, len(payment_schedule)):
                fixing_time = payment_schedule[i - 1]
                payment_time = payment_schedule[i]
                accrual_period = payment_time - fixing_time
                k_bond = 1 + strike * accrual_period
                put_price = self.zero_bond_put(fixing_time, payment_time, 1 / k_bond, mc=False)
                cap_value += k_bond * put_price

        return notional * cap_value

    def floor(self, payment_schedule: np.ndarray, notional: float, strike: float, mc: bool = False) -> float:
        self.validate_positive(notional, "notional")
        self.validate_positive(strike, "strike")
        
        floor_value = 0.0
        if mc:
            for i in range(1, len(payment_schedule)):
                fixing_time = payment_schedule[i - 1]
                payment_time = payment_schedule[i]
                accrual_period = payment_time - fixing_time
                forward_rate = self.curve_sim.fwd_rate(fixing_time, fixing_time, payment_time, fwd_measure=True)
                payoff = accrual_period * np.maximum(strike - forward_rate, 0)
                discount_payment = self.model.discount(payment_time)
                floor_value += discount_payment * np.mean(payoff)
        else:
            for i in range(1, len(payment_schedule)):
                fixing_time = payment_schedule[i - 1]
                payment_time = payment_schedule[i]
                accrual_period = payment_time - fixing_time
                k_bond = 1 + strike * accrual_period
                call_price = self.zero_bond_call(fixing_time, payment_time, 1 / k_bond, mc=False)
                floor_value += k_bond * call_price

        return notional * floor_value

    def swap(self, payment_schedule: np.ndarray, notional: float, fixed_rate: float, payer: bool = True, mc: bool = False) -> float:
        self.validate_positive(notional, "notional")
        self.validate_positive(fixed_rate, "fixed_rate")
        
        direction = 1.0 if payer else -1.0
        annuity = 0.0
        for i in range(1, len(payment_schedule)):
            accrual_period = payment_schedule[i] - payment_schedule[i - 1]
            discount_payment = self.model.discount(payment_schedule[i])
            annuity += accrual_period * discount_payment

        fixed_leg_pv = annuity * fixed_rate
        floating_leg_pv = 0.0

        if mc:
            for i in range(1, len(payment_schedule)):
                fixing_time = payment_schedule[i - 1]
                payment_time = payment_schedule[i]
                accrual_period = payment_time - fixing_time
                discount_payment = self.model.discount(payment_time)
                forward_rate = self.curve_sim.fwd_rate(fixing_time, fixing_time, payment_time, fwd_measure=True)
                floating_leg_pv += discount_payment * accrual_period * np.mean(forward_rate)
        else:
            floating_leg_pv = self.model.discount(payment_schedule[0]) - self.model.discount(payment_schedule[-1])

        swap_value = notional * direction * (floating_leg_pv - fixed_leg_pv)
        return swap_value

    def _jams_root(self, option_expiry: float, payment_schedule: np.ndarray, fixed_rate: float, critical_rate: float) -> float:
        root = 0.0
        bond_price_final = 0.0
        for i in range(1, len(payment_schedule)):
            fixing_time = payment_schedule[i - 1]
            payment_time = payment_schedule[i]
            accrual_period = payment_time - fixing_time
            bond_sens = self.model.rate_sens(option_expiry, payment_time)
            bond_adj = self.model.bond_adj_factor(option_expiry, payment_time)
            bond_price_i = bond_adj * np.exp(-bond_sens * critical_rate)
            root += accrual_period * fixed_rate * bond_price_i
            if i == len(payment_schedule) - 1:
                bond_price_final = bond_price_i

        root = root - (1 - bond_price_final)
        return root

    def _find_rstar(self, option_expiry: float, payment_schedule: np.ndarray, fixed_rate: float, lower_bound: float = -5.0, upper_bound: float = 5.0) -> float:
        root_func = lambda rate: self._jams_root(option_expiry, payment_schedule, fixed_rate, rate)
        
        f_lower = root_func(lower_bound)
        f_upper = root_func(upper_bound)
        
        if f_lower * f_upper > 0:
            lower_bound, upper_bound = -10.0, 10.0
            f_lower = root_func(lower_bound)
            f_upper = root_func(upper_bound)
            if f_lower * f_upper > 0:
                raise ValueError(f"Root not bracketed in [{lower_bound}, {upper_bound}]. Check fixed_rate={fixed_rate}.")
        
        return brentq(root_func, lower_bound, upper_bound, xtol=1e-12)

    def swaption(self, payment_schedule: np.ndarray, notional: float, fixed_rate: float, payer: bool = True, mc: bool = False) -> float:
        self.validate_positive(notional, "notional")
        self.validate_positive(fixed_rate, "fixed_rate")
        
        direction = 1.0 if payer else -1.0
        option_expiry = payment_schedule[0]
        swap_maturity = payment_schedule[-1]

        if mc:
            short_rate_expiry = self.curve_sim.sim.sim_short_rate_direct_fwd(option_expiry)
            bond_adj_final = self.model.bond_adj_factor(option_expiry, swap_maturity)
            bond_sens_final = self.model.rate_sens(option_expiry, swap_maturity)
            bond_price_final = bond_adj_final * np.exp(-bond_sens_final * short_rate_expiry)
            floating_leg_value = 1.0 - bond_price_final
            fixed_leg_value = 0.0

            for i in range(1, len(payment_schedule)):
                payment_time = payment_schedule[i]
                accrual_period = payment_time - payment_schedule[i - 1]
                bond_adj_i = self.model.bond_adj_factor(option_expiry, payment_time)
                bond_sens_i = self.model.rate_sens(option_expiry, payment_time)
                bond_price_i = bond_adj_i * np.exp(-bond_sens_i * short_rate_expiry)
                fixed_leg_value += accrual_period * fixed_rate * bond_price_i

            disc_expiry = self.model.discount(option_expiry)
            swaption_value = disc_expiry * notional * np.mean(np.maximum(direction * (floating_leg_value - fixed_leg_value), 0))

        else:
            critical_rate = self._find_rstar(option_expiry, payment_schedule, fixed_rate)
            fixed_leg_value = 0.0
            for i in range(1, len(payment_schedule)):
                fixing_time = payment_schedule[i - 1]
                payment_time = payment_schedule[i]
                accrual_period = payment_time - fixing_time
                bond_sens = self.model.rate_sens(option_expiry, payment_time)
                bond_adj = self.model.bond_adj_factor(option_expiry, payment_time)
                strike_bond = bond_adj * np.exp(-bond_sens * critical_rate)
                option = self.zero_bond_put(option_expiry, payment_time, strike_bond, mc=False) if payer else self.zero_bond_call(option_expiry, payment_time, strike_bond, mc=False)
                fixed_leg_value += accrual_period * fixed_rate * option

            bond_sens_final = self.model.rate_sens(option_expiry, swap_maturity)
            bond_adj_final = self.model.bond_adj_factor(option_expiry, swap_maturity)
            strike_final = bond_adj_final * np.exp(-bond_sens_final * critical_rate)
            floating_leg_value = self.zero_bond_put(option_expiry, swap_maturity, strike_final, mc=False) if payer else self.zero_bond_call(option_expiry, swap_maturity, strike_final, mc=False)
            swaption_value = notional * (floating_leg_value + fixed_leg_value)

        return swaption_value

    def coupon_bond(self, payment_schedule: np.ndarray, coupon_rate: float, notional: float) -> float:
        self.validate_positive(coupon_rate, "coupon_rate")
        self.validate_positive(notional, "notional")
        
        bond_price = 0.0
        for i in range(len(payment_schedule)):
            curr_time = payment_schedule[i]
            previous_time = payment_schedule[i - 1] if i > 0 else 0.0
            accrual_period = curr_time - previous_time
            disc_payment = self.curve.discount(curr_time)
            coupon_cashflow = notional * coupon_rate * accrual_period

            if i == len(payment_schedule) - 1:
                coupon_cashflow += notional

            bond_price += coupon_cashflow * disc_payment

        return bond_price

    def floating_rate_note(self, payment_schedule: np.ndarray, notional: float) -> float:
        self.validate_positive(notional, "notional")
        
        disced_coupons = self.swap(payment_schedule, notional, fixed_rate=0.0, payer=False, mc=False)
        disced_notional = notional * self.model.discount(payment_schedule[-1])
        frn_price = disced_coupons + disced_notional
        return frn_price

    def _jams_root_bond(self, option_expiry: float, payment_schedule: np.ndarray, coupon_rate: float, notional: float, strike_price: float, critical_rate: float) -> float:
        bond_price = 0.0
        for i in range(len(payment_schedule)):
            current_time = payment_schedule[i]
            previous_time = payment_schedule[i - 1] if i > 0 else 0.0
            accrual_period = current_time - previous_time
            bond_sens = self.model.rate_sens(option_expiry, current_time)
            bond_adj = self.model.bond_adj_factor(option_expiry, current_time)
            bond_price_i = bond_adj * np.exp(-bond_sens * critical_rate)
            coupon_cashflow = notional * coupon_rate * accrual_period
            if i == len(payment_schedule) - 1:
                coupon_cashflow += notional
            bond_price += coupon_cashflow * bond_price_i

        return bond_price - strike_price

    def _find_rstar_bond(self, option_expiry: float, payment_schedule: np.ndarray, coupon_rate: float, notional: float, strike_price: float, lower_bound: float = -5.0, upper_bound: float = 5.0) -> float:
        root_func = lambda rate: self._jams_root_bond(option_expiry, payment_schedule, coupon_rate, notional, strike_price, rate)
        
        f_lower = root_func(lower_bound)
        f_upper = root_func(upper_bound)
        
        if f_lower * f_upper > 0:
            lower_bound, upper_bound = -10.0, 10.0
            f_lower = root_func(lower_bound)
            f_upper = root_func(upper_bound)
            if f_lower * f_upper > 0:
                raise ValueError(f"Root not bracketed in [{lower_bound}, {upper_bound}]. Check strike_price={strike_price}.")
        
        return brentq(root_func, lower_bound, upper_bound, xtol=1e-12)

    def bond_option(self, option_expiry: float, payment_schedule: np.ndarray, coupon_rate: float, strike: float, notional: float, call: bool = True, mc: bool = False) -> float:
        self.validate_times(option_expiry, payment_schedule[-1])
        self.validate_positive(coupon_rate, "coupon_rate")
        self.validate_positive(strike, "strike")
        self.validate_positive(notional, "notional")
        
        if mc:
            bond_price_dist = self.curve_sim.coupon_bond(option_expiry, payment_schedule, coupon_rate, notional)
            disc_factor = self.model.discount(option_expiry)
            option_value = disc_factor * np.mean(
                np.maximum(bond_price_dist - strike, 0) if call else np.maximum(strike - bond_price_dist, 0)
            )

        else:
            critical_rate = self._find_rstar_bond(option_expiry, payment_schedule, coupon_rate, notional, strike)
            option_value = 0.0

            for i in range(len(payment_schedule)):
                curr_time = payment_schedule[i]
                prev_time = payment_schedule[i - 1] if i > 0 else 0.0
                accrual_period = curr_time - prev_time
                bond_sens = self.model.rate_sens(option_expiry, curr_time)
                bond_adj = self.model.bond_adj_factor(option_expiry, curr_time)
                k_bond = bond_adj * np.exp(-bond_sens * critical_rate)
                option = self.zero_bond_call(option_expiry, curr_time, k_bond, mc=False) if call else self.zero_bond_put(option_expiry, curr_time, k_bond, mc=False)
                coupon_cashflow = notional * coupon_rate * accrual_period
                if i == len(payment_schedule) - 1:
                    coupon_cashflow += notional
                option_value += coupon_cashflow * option

        return option_value

if __name__ == "__main__":
    print("pricer works")
