"""Interest Rate Derivatives Pricing"""
#FULLY DONE

import numpy as np
from scipy.stats import norm
from scipy.optimize import brentq
from hw_model import hwCurveBuilder


class hwPricer:
    def __init__(self, curve, n_paths=10**5, n_steps=252, seed=2025, hw_params=None):
        self.curve = curve
        self.curve_sim = hwCurveBuilder(curve, params=hw_params, n_paths=n_paths, n_steps=n_steps, seed=seed)
        self.model = self.curve_sim.model

    def set_sim(self, n_paths=None, n_steps=None, seed=None):
        if n_paths is not None:
            self.curve_sim.sim.n_paths = int(n_paths)
        if n_steps is not None:
            self.curve_sim.sim.n_steps = int(n_steps)
        if seed is not None:
            np.random.seed(seed)

    def zero_bond_put(self, option_expiry, bond_maturity, k, mc=False):
        if option_expiry == 0:
            bond_price = self.model.discount(bond_maturity)
            return max(k - bond_price, 0)

        if mc:
            df = self.model.discount(option_expiry)
            bond_price_sim = self.curve_sim.zero_coupon_bond(option_expiry, bond_maturity, fwd_measure=True)
            payoff = np.maximum(k - bond_price_sim, 0)
            option_value = np.mean(df * payoff)
        else:
            sigma = self.model.parameters['sigma']
            a = self.model.parameters['a']
            bond_sens = self.model.rate_sens(option_expiry, bond_maturity)
            bond_price_maturity = self.model.discount(bond_maturity)
            bond_price_expiry = self.model.discount(option_expiry)
            vol_bond = sigma * np.sqrt((1 - np.exp(-2 * a * option_expiry)) / (2 * a)) * bond_sens
            h = (1 / vol_bond) * np.log(bond_price_maturity / (k * bond_price_expiry)) + 0.5 * vol_bond
            option_value = k * bond_price_expiry * norm.cdf(-h + vol_bond) - bond_price_maturity * norm.cdf(-h)

        return option_value

    def zero_bond_call(self, option_expiry, bond_maturity, k, mc=False):
        if mc:
            df = self.model.discount(option_expiry)
            bond_price_sim = self.curve_sim.zero_coupon_bond(option_expiry, bond_maturity, fwd_measure=True)
            payoff = np.maximum(bond_price_sim - k, 0)
            option_value = np.mean(df * payoff)
        else:
            sigma = self.model.parameters['sigma']
            a = self.model.parameters['a']
            bond_sens = self.model.rate_sens(option_expiry, bond_maturity)
            bond_price_maturity = self.model.discount(bond_maturity)
            bond_price_expiry = self.model.discount(option_expiry)
            vol_bond = sigma * np.sqrt((1 - np.exp(-2 * a * option_expiry)) / (2 * a)) * bond_sens
            h = (1 / vol_bond) * np.log(bond_price_maturity / (k * bond_price_expiry)) + 0.5 * vol_bond
            option_value = bond_price_maturity * norm.cdf(h) - k * bond_price_expiry * norm.cdf(h - vol_bond)

        return option_value

    def caplet(self, fixing_time, payment_time, notional, k, method='js'):
        accrual_period = payment_time - fixing_time
        k_bond = 1 + k * accrual_period

        if method == 'mc':
            forward_rate = self.curve_sim.inst_fwd_rate(fixing_time, fixing_time, payment_time)
            payoff = accrual_period * np.maximum(forward_rate - k, 0)
            discount_payment = self.model.discount(payment_time)
            caplet_value = discount_payment * np.mean(payoff)

        elif method == 'js':
            put_price = self.zero_bond_put(fixing_time, payment_time, 1 / k_bond)
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

        return notional * caplet_value

    def cap(self, payment_schedule, notional, k, mc=False):
        cap_value = 0
        if mc:
            for i in range(1, len(payment_schedule)):
                fixing_time = payment_schedule[i - 1]
                payment_time = payment_schedule[i]
                accrual_period = payment_time - fixing_time

                forward_rate = self.curve_sim.fwd_rate(fixing_time, fixing_time, payment_time, fwd_measure=True)

                payoff = accrual_period * np.maximum(forward_rate - k, 0)
                discount_payment = self.model.discount(payment_time)
                cap_value += discount_payment * np.mean(payoff)
        else:
            for i in range(1, len(payment_schedule)):
                fixing_time = payment_schedule[i - 1]
                payment_time = payment_schedule[i]
                accrual_period = payment_time - fixing_time
                k_bond = 1 + k * accrual_period
                put_price = self.zero_bond_put(fixing_time, payment_time, 1 / k_bond)
                cap_value += k_bond * put_price

        return notional * cap_value

    def floor(self, payment_schedule, notional, k, mc=False):
        floor_value = 0
        if mc:
            for i in range(1, len(payment_schedule)):
                fixing_time = payment_schedule[i - 1]
                payment_time = payment_schedule[i]
                accrual_period = payment_time - fixing_time
                forward_rate = self.curve_sim.fwd_rate(fixing_time, fixing_time, payment_time, fwd_measure=True)
                payoff = accrual_period * np.maximum(k - forward_rate, 0)
                discount_payment = self.model.discount(payment_time)
                floor_value += discount_payment * np.mean(payoff)
        else:
            for i in range(1, len(payment_schedule)):
                fixing_time = payment_schedule[i - 1]
                payment_time = payment_schedule[i]
                accrual_period = payment_time - fixing_time
                k_bond = 1 + k * accrual_period
                call_price = self.zero_bond_call(fixing_time, payment_time, 1 / k_bond)
                floor_value += k_bond * call_price

        return notional * floor_value

    def swap(self, payment_schedule, notional, fixed_rate, payer=True, mc=False):
        direction = 1 if payer else -1
        annuity = 0
        for i in range(1, len(payment_schedule)):
            accrual_period = payment_schedule[i] - payment_schedule[i - 1]
            discount_payment = self.model.discount(payment_schedule[i])
            annuity += accrual_period * discount_payment

        fixed_leg_pv = annuity * fixed_rate
        floating_leg_pv = 0

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

    def swaption(self, payment_schedule, notional, fixed_rate, payer=True, mc=False):
        direction = 1 if payer else -1
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
            critical_rate = self.find_rstar(option_expiry, payment_schedule, fixed_rate)
            fixed_leg_value = 0
            for i in range(1, len(payment_schedule)):
                fixing_time = payment_schedule[i - 1]
                payment_time = payment_schedule[i]
                accrual_period = payment_time - fixing_time
                bond_sens = self.model.rate_sens(option_expiry, payment_time)
                bond_adj = self.model.bond_adj_factor(option_expiry, payment_time)
                strike_bond = bond_adj * np.exp(-bond_sens * critical_rate)
                option = self.zero_bond_put(option_expiry, payment_time, strike_bond) if payer else self.zero_bond_call(option_expiry, payment_time, strike_bond)
                fixed_leg_value += accrual_period * fixed_rate * option

            bond_sens_final = self.model.rate_sens(option_expiry, swap_maturity)
            bond_adj_final = self.model.bond_adj_factor(option_expiry, swap_maturity)
            strike_final = bond_adj_final * np.exp(-bond_sens_final * critical_rate)
            floating_leg_value = self.zero_bond_put(option_expiry, swap_maturity, strike_final) if payer else self.zero_bond_call(option_expiry, swap_maturity, strike_final)
            swaption_value = notional * (floating_leg_value + fixed_leg_value)

        return swaption_value

    def coupon_bond(self, payment_schedule, coupon_rate, notional):
        bond_price = 0
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

    def floating_rate_note(self, payment_schedule, notional):
        disced_coupons = self.swap(payment_schedule, notional, fixed_rate=0, payer=False, mc=False)
        disced_notional = notional * self.model.discount(payment_schedule[-1])
        frn_price = disced_coupons + disced_notional
        return frn_price

    def bond_option(self, option_expiry, payment_schedule, coupon_rate, k, notional, call=True, mc=False):
        if mc:
            bond_price_dist = self.curve_sim.coupon_bond(option_expiry, payment_schedule, coupon_rate, notional)
            disc_factor = self.model.discount(option_expiry)
            option_value = disc_factor * np.mean(
                np.maximum(bond_price_dist - k, 0) if call else np.maximum(k - bond_price_dist, 0)
            )

        else:
            critical_rate = self.find_rstar_bond(option_expiry, payment_schedule, coupon_rate, notional, k)
            option_value = 0

            for i in range(len(payment_schedule)):
                curr_time = payment_schedule[i]
                prev_time = payment_schedule[i - 1] if i > 0 else 0.0
                accrual_period = curr_time - prev_time

                bond_sens = self.model.rate_sens(option_expiry, curr_time)
                bond_adj = self.model.bond_adj_factor(option_expiry, curr_time)
                k_bond = bond_adj * np.exp(-bond_sens * critical_rate)

                option = self.zero_bond_call(option_expiry, curr_time, k_bond) if call else self.zero_bond_put(option_expiry, curr_time, k_bond)

                coupon_cashflow = notional * coupon_rate * accrual_period
                if i == len(payment_schedule) - 1:
                    coupon_cashflow += notional

                option_value += coupon_cashflow * option

        return option_value

    def jams_root(self, option_expiry, payment_schedule, fixed_rate, critical_rate):
        root = 0
        bond_price_final = 0
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

    def find_rstar(self, option_expiry, payment_schedule, fixed_rate, lower_bound=-5.0, upper_bound=5.0):
        root_func = lambda rate: self.jams_root(option_expiry, payment_schedule, fixed_rate, rate)
        try:
            critical_rate = brentq(root_func, lower_bound, upper_bound, xtol=1e-12)
        except ValueError:
            critical_rate = brentq(root_func, -10.0, 10.0, xtol=1e-12)
        return critical_rate

    def jams_root_bond(self, option_expiry, payment_schedule, coupon_rate, notional, k_price, critical_rate):
        bond_price = 0

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

        return bond_price - k_price

    def find_rstar_bond(self, option_expiry, payment_schedule, coupon_rate, notional, k_price, lower_bound=-5.0, upper_bound=5.0):
        root_func = lambda rate: self.jams_root_bond(option_expiry, payment_schedule, coupon_rate, notional, k_price, rate)
        try:
            critical_rate = brentq(root_func, lower_bound, upper_bound, xtol=1e-12)
        except ValueError:
            critical_rate = brentq(root_func, -10.0, 10.0, xtol=1e-12)
        return critical_rate
