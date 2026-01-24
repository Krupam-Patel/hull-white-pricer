"""Interest Rate Derivatives Pricing"""
# Fully done 1/21/2026
# Commenting done 1/24/2026

# I tried putting comments to explain my thought process for everything in the code
# If something is still fuzzy, there is a typo, or there is a better way to do something, please dm on linkedin!
#https://www.linkedin.com/in/krupam-patel/


import logging
import numpy as np
from scipy.stats import norm
from scipy.optimize import brentq

from hw_model import hwCurveBuilder

logger = logging.getLogger(__name__)


class hwPricer:
    def __init__(self, curve, n_paths: int = 10**5, n_steps: int = 252, seed: int = 2025, hw_params: dict | None = None):
        self.curve = curve
        self.curve_sim = hwCurveBuilder(curve, params=hw_params, n_paths=n_paths, n_steps=n_steps, seed=seed)
        self.model = self.curve_sim.model

    def set_sim(self, n_paths=None, n_steps=None, seed=None):
        # Updates number of Monte Carlo paths
        if n_paths is not None:
            if not isinstance(n_paths, (int, float)) or n_paths <= 0:
                raise ValueError(f"n_paths got to be positive, got {n_paths}")
            self.curve_sim.sim.n_paths = int(n_paths)

        # Updates number of time steps per path
        if n_steps is not None:
            if not isinstance(n_steps, (int, float)) or n_steps <= 0:
                raise ValueError(f"n_steps got to be positive, got {n_steps}")
            self.curve_sim.sim.n_steps = int(n_steps)

        # Resets random seed for consistant results
        if seed is not None:
            if not isinstance(seed, (int, float)):
                raise ValueError(f"seed got to be an integer, got {seed}")
            np.random.seed(int(seed))

    def validate_times(self, option_expiry: float, bond_maturity: float) -> None:
        # Sanity checks for option expiry and bond maturity
        if not isinstance(option_expiry, (int, float)) or option_expiry < 0:
            raise ValueError(f"option_expiry must be non-negative float, got {option_expiry}")
        if not isinstance(bond_maturity, (int, float)) or bond_maturity < 0:
            raise ValueError(f"bond_maturity must be non-negative float, got {bond_maturity}")
        if bond_maturity < option_expiry:
            # Makes sure option cannot expire after the underlying bond matures
            raise ValueError(f"bond_maturity ({bond_maturity}) must be >= option_expiry ({option_expiry})")

    def validate_positive(self, value: float, name: str) -> None:
        # Makes sure notionals, strikes, rates is positive
        if not isinstance(value, (int, float)) or value <= 0:
            raise ValueError(f"{name} must be positive, got {value}")

    def price_zero_bond_put_mc(self, option_expiry: float, bond_maturity: float, k: float) -> float:
        # Discount factor from today to option expiry under the risk‑neutral measure
        df = self.model.discount(option_expiry)
        # Simulates zero-coupon bond price at option expiry in forward measure
        bond_price_sim = self.curve_sim.zero_coupon_bond(option_expiry, bond_maturity, fwd_measure=True)
        # Pathwise puts payoff on the bond
        payoff = np.maximum(k - bond_price_sim, 0)
        logger.debug("MC Zero Bond Put Payoff: %s", payoff)

        # PV = discounted expected payoff
        return np.mean(df * payoff)

    def price_zero_bond_put_analytical(self, option_expiry: float, bond_maturity: float, k: float) -> float:
        # Hull–White parameters
        sigma = self.model.parameters["sigma"]
        a = self.model.parameters["a"]
        # Sensitivity of ZCB price to short rate at expiry (B(t,T) in HW notation)
        bond_sens = self.model.rate_sens(option_expiry, bond_maturity)
        bond_price_maturity = self.model.discount(bond_maturity)
        bond_price_expiry = self.model.discount(option_expiry)
        # Bond price volatility over from 0 through option expiry
        vol_bond = sigma * np.sqrt((1 - np.exp(-2 * a * option_expiry)) / (2 * a)) * bond_sens
        # Standard normal deviate equation used in closed-form bond option
        h = (1 / vol_bond) * np.log(bond_price_maturity / (k * bond_price_expiry)) + 0.5 * vol_bond

        # Put price for Closed-form zero-coupon bond equation
        return k * bond_price_expiry * norm.cdf(-h + vol_bond) - bond_price_maturity * norm.cdf(-h)

    def zero_bond_put(self, option_expiry: float, bond_maturity: float, k: float, mc: bool = False) -> float:
        # Zero-coupon bond put pricer (MC or analytical)
        self.validate_times(option_expiry, bond_maturity)
        self.validate_positive(k, "strike")
        # Immediate exercise at t=0 (no optionality)
        if option_expiry == 0.0:
            bond_price = self.model.discount(bond_maturity)
            return max(k - bond_price, 0.0)
        # Monte Carlo route
        if mc:  
            return self.price_zero_bond_put_mc(option_expiry, bond_maturity, k)

        # Analytical closed-form route 
        return self.price_zero_bond_put_analytical(option_expiry, bond_maturity, k)

    def price_zero_bond_call_mc(self, option_expiry: float, bond_maturity: float, k: float) -> float:
        # MC pricing for zero-coupon bond call option
        if k <= 0:
            raise ValueError(f"strike must be positive, got {k}")
        df = self.model.discount(option_expiry)
        # Simulate bond price at option expiry
        bond_price_sim = self.curve_sim.zero_coupon_bond(option_expiry, bond_maturity, fwd_measure=True)
        # Pathwise call payoff
        payoff = np.maximum(bond_price_sim - k, 0.0)

        # PV as discounted average payoff
        return float(np.mean(df * payoff))

    def price_zero_bond_call_analytical(self, option_expiry: float, bond_maturity: float, k: float) -> float:
        # Analytical Hull–White formula for zero-coupon bond call
        if k <= 0:
            raise ValueError(f"strike must be positive, got {k}")
        sigma = self.model.parameters["sigma"]
        a = self.model.parameters["a"]
        bond_sens = self.model.rate_sens(option_expiry, bond_maturity)
        bond_price_maturity = self.model.discount(bond_maturity)
        bond_price_expiry = self.model.discount(option_expiry)
        vol_bond = sigma * np.sqrt((1.0 - np.exp(-2.0 * a * option_expiry)) / (2.0 * a)) * bond_sens
        h = (1.0 / vol_bond) * np.log(bond_price_maturity / (k * bond_price_expiry)) + 0.5 * vol_bond

        # Standard bond call expression in HW model
        return float(bond_price_maturity * norm.cdf(h) - k * bond_price_expiry * norm.cdf(h - vol_bond))

    def zero_bond_call(self, option_expiry: float, bond_maturity: float, k: float, mc: bool = False) -> float:
        # Zero-coupon bond call pricer
        self.validate_times(option_expiry, bond_maturity)
        self.validate_positive(k, "strike")

        if mc:
            return self.price_zero_bond_call_mc(option_expiry, bond_maturity, k)

        return self.price_zero_bond_call_analytical(option_expiry, bond_maturity, k)

    def caplet(self, fixing_time: float, payment_time: float, notional: float, k: float, method: str = "js") -> float:
        # Price a single caplet using different methods (MC, Jamshidian, closed-form)
        self.validate_times(fixing_time, payment_time)
        self.validate_positive(notional, "notional")
        self.validate_positive(k, "strike")

        accrual_period = payment_time - fixing_time
        # Strike on the corresponding bond
        k_bond = 1.0 + k * accrual_period

        if method == "mc":  # Monte Carlo
            # Simulate instantaneous forward rate over [fixing, payment]
            fwd_rate = self.curve_sim.inst_fwd_rate(fixing_time, fixing_time, payment_time)
            # Caplet payoff in terms of the rate
            payoff = accrual_period * np.maximum(fwd_rate - k, 0.0)
            disc_payment = self.model.discount(payment_time)
            caplet_value = disc_payment * np.mean(payoff)

        elif method == "js":  # Jamshidian's Split
            # Use bond put with Jamshidian decomposition
            put_price = self.zero_bond_put(fixing_time, payment_time, 1.0 / k_bond, mc=False)
            caplet_value = k_bond * put_price

        elif method == "cf":  # Closed-Form
            # Closed-form caplet formula in Hull–White using bond volatility
            sigma = self.model.parameters["sigma"]
            mean_reversion = self.model.parameters["a"]
            bond_sens = self.model.rate_sens(fixing_time, payment_time)
            disc_payment = self.model.discount(payment_time)
            disc_fixing = self.model.discount(fixing_time)
            # Volatility of discount bond over 0 and fixing_time
            vol_bond = (sigma * np.sqrt((1.0 - np.exp(-2.0 * mean_reversion * fixing_time)) / (2.0 * mean_reversion)) * bond_sens)
            h = (1.0 / vol_bond) * np.log(disc_payment * k_bond / disc_fixing) + 0.5 * vol_bond
            # Analytical caplet price in terms of ZCB options
            caplet_value = disc_fixing * norm.cdf(-h + vol_bond) - k_bond * disc_payment * norm.cdf(-h)
        else:
            # Making sure model is not trippin 
            raise ValueError(f"method must be 'mc', 'js', or 'cf', got {method}")

        return notional * caplet_value

    def caplet_pv(self, fixing_time: float, payment_time: float, k: float, mc: bool) -> float:
        # PV of a unit-notional caplet
        accrual_period = payment_time - fixing_time
        if mc: # MC valuation using forward rate simulation
            fwd_rate = self.curve_sim.fwd_rate(fixing_time, fixing_time, payment_time, fwd_measure=True)
            payoff = accrual_period * np.maximum(fwd_rate - k, 0.0)
            disc_payment = self.model.discount(payment_time)

            return disc_payment * np.mean(payoff)
        else: # Jamshidian decomposition via bond put
            k_bond = 1.0 + k * accrual_period
            put_price = self.zero_bond_put(fixing_time, payment_time, 1.0 / k_bond, mc=False)

            return k_bond * put_price

    def cap(self, payment_schedule: np.ndarray, notional: float, k: float, mc: bool = False) -> float:
        # Price an interest rate cap as sum of caplets
        self.validate_positive(notional, "notional")
        self.validate_positive(k, "strike")

        # Sum unit-notional caplet PVs across payment dates
        cap_value = sum(self.caplet_pv(payment_schedule[i - 1], payment_schedule[i], k, mc)
            for i in range(1, len(payment_schedule))
        )

        return notional * cap_value

    def floor(self, payment_schedule: np.ndarray, notional: float, k: float, mc: bool = False) -> float:
        # Price an interest rate floor
        self.validate_positive(notional, "notional")
        self.validate_positive(k, "strike")

        floor_value = 0.0
        if mc: # MC valuation by simulating forward rates
            for i in range(1, len(payment_schedule)):
                fixing_time = payment_schedule[i - 1]
                payment_time = payment_schedule[i]
                accrual_period = payment_time - fixing_time
                fwd_rate = self.curve_sim.fwd_rate(
                    fixing_time, fixing_time, payment_time, fwd_measure=True
                )
                payoff = accrual_period * np.maximum(k - fwd_rate, 0.0)
                disc_payment = self.model.discount(payment_time)
                floor_value += disc_payment * np.mean(payoff)
        else:
            # Analytical valuation via call options (Jamshidian)
            for i in range(1, len(payment_schedule)):
                fixing_time = payment_schedule[i - 1]
                payment_time = payment_schedule[i]
                accrual_period = payment_time - fixing_time
                k_bond = 1.0 + k * accrual_period
                call_price = self.zero_bond_call(fixing_time, payment_time, 1.0 / k_bond, mc=False)
                floor_value += k_bond * call_price

        return notional * floor_value

    def swap(self, payment_schedule: np.ndarray, notional: float, fixed_rate: float, payer: bool = True, mc: bool = False) -> float:
        # Price a plain-vanilla interest rate swap
        self.validate_positive(notional, "notional")
        self.validate_positive(fixed_rate, "fixed_rate")

        direction = 1.0 if payer else -1.0
        annuity = 0.0
        for i in range(1, len(payment_schedule)):
            accrual_period = payment_schedule[i] - payment_schedule[i - 1]
            disc_payment = self.model.discount(payment_schedule[i])
            annuity += accrual_period * disc_payment

        # Fixed leg PV for given fixed rate
        fixed_leg_pv = annuity * fixed_rate
        floating_leg_pv = 0.0

        if mc: # MC valuation of floating leg via simulated forward rates
            for i in range(1, len(payment_schedule)):
                fixing_time = payment_schedule[i - 1]
                payment_time = payment_schedule[i]
                accrual_period = payment_time - fixing_time
                disc_payment = self.model.discount(payment_time)
                fwd_rate = self.curve_sim.fwd_rate(fixing_time, fixing_time, payment_time, fwd_measure=True)
                floating_leg_pv += disc_payment * accrual_period * np.mean(fwd_rate)
        else: # Analytical floating leg PV using bond parity (aka using P(0,T0) - P(0,Tn))
            floating_leg_pv = self.model.discount(payment_schedule[0]) - self.model.discount(payment_schedule[-1])
        # Swap value from perspective of payer/receiver
        swap_value = notional * direction * (floating_leg_pv - fixed_leg_pv)

        return swap_value

    def jams_root(self, option_expiry: float, payment_schedule: np.ndarray, fixed_rate: float, critical_rate: float) -> float:
        # Root function for Jamshidian decomposition on a swap (finds critical short rate)
        root = 0.0
        bond_price_final = 0.0
        for i in range(1, len(payment_schedule)):
            fixing_time = payment_schedule[i - 1]
            payment_time = payment_schedule[i]
            accrual_period = payment_time - fixing_time
            # Sensitivity and adjustment factor for ZCB under HW at option expiry
            bond_sens = self.model.rate_sens(option_expiry, payment_time)
            bond_adj = self.model.bond_adj_factor(option_expiry, payment_time)
            # Bond price as function of critical rate 
            bond_price_i = bond_adj * np.exp(-bond_sens * critical_rate)
            # PV of fixed-leg cashflow
            root += accrual_period * fixed_rate * bond_price_i
            if i == len(payment_schedule) - 1:
                # Track last bond price for floating leg (par swap)
                bond_price_final = bond_price_i

        root = root - (1.0 - bond_price_final)

        return root

    def find_rstar(self, option_expiry: float, payment_schedule: np.ndarray, fixed_rate: float, lower_bound: float = -5.0, upper_bound: float = 5.0) -> float:
        # Solve for Jamshidian critical rate for a swaption
        root_func = lambda rate: self.jams_root(option_expiry, payment_schedule, fixed_rate, rate)
        # Evaluate function at initial bounds
        f_lower = root_func(lower_bound)
        f_upper = root_func(upper_bound)
        # If root not bracketed, widens the search interval
        if f_lower * f_upper > 0:
            lower_bound, upper_bound = -10.0, 10.0
            f_lower = root_func(lower_bound)
            f_upper = root_func(upper_bound)
            if f_lower * f_upper > 0:
                # Fails if theres still no sign change
                raise ValueError(f"Root not bracketed in [{lower_bound}, {upper_bound}]. Check fixed rate={fixed_rate}.")

        return brentq(root_func, lower_bound, upper_bound, xtol=1e-12)

    def swaption(self, payment_schedule: np.ndarray, notional: float, fixed_rate: float, payer: bool = True, mc: bool = False) -> float:
        # Price a European swaption on a vanilla swap
        self.validate_positive(notional, "notional")
        self.validate_positive(fixed_rate, "fixed_rate")

        # Direction: payer swaption = right to enter payer swap
        direction = 1.0 if payer else -1.0
        option_expiry = payment_schedule[0]
        swap_maturity = payment_schedule[-1]

        if mc: # MC valuation using distribution of short rate at option expiry
            short_rate_expiry = self.curve_sim.sim.sim_short_rate_direct_fwd(option_expiry)
            bond_adj_final = self.model.bond_adj_factor(option_expiry, swap_maturity)
            bond_sens_final = self.model.rate_sens(option_expiry, swap_maturity)
            # Final bond price as function of simulated short rate
            bond_price_final = bond_adj_final * np.exp(-bond_sens_final * short_rate_expiry)
            # Floating leg PV at expiry (par swap identity)
            floating_leg_value = 1.0 - bond_price_final
            fixed_leg_value = 0.0

            # Fixed leg PV at expiry across all cashflows
            for i in range(1, len(payment_schedule)):
                payment_time = payment_schedule[i]
                accrual_period = payment_time - payment_schedule[i - 1]
                bond_adj_i = self.model.bond_adj_factor(option_expiry, payment_time)
                bond_sens_i = self.model.rate_sens(option_expiry, payment_time)
                bond_price_i = bond_adj_i * np.exp(-bond_sens_i * short_rate_expiry)
                fixed_leg_value += accrual_period * fixed_rate * bond_price_i

            # Discount option payoff from expiry to today
            disc_expiry = self.model.discount(option_expiry)
            swaption_value = disc_expiry * notional * np.mean(np.maximum(direction * (floating_leg_value - fixed_leg_value), 0))

        else: # Analytical Jamshidian decomposition for swaption
            critical_rate = self.find_rstar(option_expiry, payment_schedule, fixed_rate)
            fixed_leg_value = 0.0
            for i in range(1, len(payment_schedule)):
                fixing_time = payment_schedule[i - 1]
                payment_time = payment_schedule[i]
                accrual_period = payment_time - fixing_time
                bond_sens = self.model.rate_sens(option_expiry, payment_time)
                bond_adj = self.model.bond_adj_factor(option_expiry, payment_time)
                # Strike bond defined at the strike rate
                strike_bond = bond_adj * np.exp(-bond_sens * critical_rate)
                # Option on the bond: put for payer, call for receiver
                option = (self.zero_bond_put(option_expiry, payment_time, strike_bond, mc=False)
                    if payer
                    else self.zero_bond_call(option_expiry, payment_time, strike_bond, mc=False))
                # Contribution of each fixed cashflow to swaption value
                fixed_leg_value += accrual_period * fixed_rate * option

            # Floating leg is replicated by last bond option
            bond_sens_final = self.model.rate_sens(option_expiry, swap_maturity)
            bond_adj_final = self.model.bond_adj_factor(option_expiry, swap_maturity)
            strike_final = bond_adj_final * np.exp(-bond_sens_final * critical_rate)
            floating_leg_value = (self.zero_bond_put(option_expiry, swap_maturity, strike_final, mc=False)
                if payer
                else self.zero_bond_call(option_expiry, swap_maturity, strike_final, mc=False))
            swaption_value = notional * (floating_leg_value + fixed_leg_value)

        return swaption_value

    def coupon_bond(self, payment_schedule: np.ndarray, coupon_rate: float, notional: float) -> float:
        # Price a fixed-coupon bond by discounting cashflows on the input curve
        self.validate_positive(coupon_rate, "coupon_rate")
        self.validate_positive(notional, "notional")

        bond_price = 0.0
        for i in range(len(payment_schedule)):
            curr_time = payment_schedule[i]
            previous_time = payment_schedule[i - 1] if i > 0 else 0.0
            accrual_period = curr_time - previous_time
            # Discount factor from market curve (not model)
            disc_payment = self.curve.discount(curr_time)
            # Coupon cashflow for this period
            coupon_cashflow = notional * coupon_rate * accrual_period

            # Add principal repayment at final maturity
            if i == len(payment_schedule) - 1:
                coupon_cashflow += notional

            bond_price += coupon_cashflow * disc_payment

        return bond_price

    def floating_rate_note(self, payment_schedule: np.ndarray, notional: float) -> float:
        # Price a floating-rate note paying par at reset dates
        self.validate_positive(notional, "notional")

        # PV of floating coupons via swap with zero fixed rate
        disced_coupons = self.swap(payment_schedule, notional, fixed_rate=0.0, payer=False, mc=False)
        # Discounted notional repayment at maturity
        disced_notional = notional * self.model.discount(payment_schedule[-1])
        frn_price = disced_coupons + disced_notional

        return frn_price

    def jams_root_bond(self, option_expiry: float, payment_schedule: np.ndarray, coupon_rate: float, notional: float,
                       strike_price: float, critical_rate: float) -> float:
        # Root function for Jamshidian decomposition on a coupon bond option
        bond_price = 0.0
        for i in range(len(payment_schedule)):
            current_time = payment_schedule[i]
            previous_time = payment_schedule[i - 1] if i > 0 else 0.0
            accrual_period = current_time - previous_time
            bond_sens = self.model.rate_sens(option_expiry, current_time)
            bond_adj = self.model.bond_adj_factor(option_expiry, current_time)
            # Bond price at current cashflow date as function of critical rate
            bond_price_i = bond_adj * np.exp(-bond_sens * critical_rate)
            # Corresponding coupon cashflow
            coupon_cashflow = notional * coupon_rate * accrual_period
            if i == len(payment_schedule) - 1:
                # Adding principal at maturity
                coupon_cashflow += notional
            bond_price += coupon_cashflow * bond_price_i

        return bond_price - strike_price

    def find_rstar_bond(self, option_expiry: float, payment_schedule: np.ndarray, coupon_rate: float, notional: float,
                        strike_price: float, lower_bound: float = -5.0, upper_bound: float = 5.0) -> float:
        # Solve for Jamshidian critical rate for coupon bond option
        root_func = lambda rate: self.jams_root_bond(option_expiry, payment_schedule, coupon_rate, notional, strike_price, rate)

        # Evaluating in initial interval
        f_lower = root_func(lower_bound)
        f_upper = root_func(upper_bound)

        # Widening interval if necessary to bracket root
        if f_lower * f_upper > 0:
            lower_bound, upper_bound = -10.0, 10.0
            f_lower = root_func(lower_bound)
            f_upper = root_func(upper_bound)
            if f_lower * f_upper > 0:
                raise ValueError(f"Root not bracketed in [{lower_bound}, {upper_bound}]. Check strike_price={strike_price}.")

        # Using Brent solver for robustness
        return brentq(root_func, lower_bound, upper_bound, xtol=1e-12)

    def bond_option(self, option_expiry: float, payment_schedule: np.ndarray, coupon_rate: float, k: float,
                    notional: float, call: bool = True, mc: bool = False) -> float:
        # Pricing a European option on a coupon-bearing bond
        self.validate_times(option_expiry, payment_schedule[-1])
        self.validate_positive(coupon_rate, "coupon_rate")
        self.validate_positive(k, "strike")
        self.validate_positive(notional, "notional")

        if mc:
            # MC valuation: simulate coupon bond price at option expiry
            bond_price_dist = self.curve_sim.coupon_bond(option_expiry, payment_schedule, coupon_rate, notional)
            disc_factor = self.model.discount(option_expiry)
            # Use call/put payoff on simulated bond prices
            option_value = disc_factor * np.mean(np.maximum(bond_price_dist - k, 0) if call else np.maximum(k - bond_price_dist, 0))
        else:
            # Analytical Jamshidian decomposition for coupon bond option
            critical_rate = self.find_rstar_bond(option_expiry, payment_schedule, coupon_rate, notional, k)
            option_value = 0.0

            for i in range(len(payment_schedule)):
                curr_time = payment_schedule[i]
                prev_time = payment_schedule[i - 1] if i > 0 else 0.0
                accrual_period = curr_time - prev_time
                bond_sens = self.model.rate_sens(option_expiry, curr_time)
                bond_adj = self.model.bond_adj_factor(option_expiry, curr_time)
                # Striking bond at each cashflow date for strike rate
                k_bond = bond_adj * np.exp(-bond_sens * critical_rate)
                # Zero-coupon bond option for each coupon cashflow
                option = (self.zero_bond_call(option_expiry, curr_time, k_bond, mc=False)
                    if call
                    else self.zero_bond_put(option_expiry, curr_time, k_bond, mc=False))
                coupon_cashflow = notional * coupon_rate * accrual_period
                if i == len(payment_schedule) - 1:
                    coupon_cashflow += notional
                option_value += coupon_cashflow * option

        return option_value


if __name__ == "__main__":
    # Simple test when running this module directly
    logging.basicConfig(level=logging.INFO)
    logger.info("hwPricer module is good")
