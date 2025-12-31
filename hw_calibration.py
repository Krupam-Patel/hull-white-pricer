"""Calibrate Hull White model sigma with fixed a"""

#Fully Done

from scipy.optimize import minimize, brentq
from scipy.stats import norm
import numpy as np
import pandas as pd


class hwCalibrator:
    def __init__(self, pricer, mar_data_df, weight_scheme='Relative', mean_rever=False):
        self.pricer = pricer
        self.model = pricer.model
        self.mar_data_df = mar_data_df
        self.weight_scheme = weight_scheme
        self.mean_rever = mean_rever
        self.cali_hist = []

    def rate_to_decimal(self, rate_input):
        return rate_input / 100.0 if rate_input and rate_input > 0.5 else rate_input

    def objective_error(self, params_vector):
        if self.mean_rever:
            mean_rever_param = params_vector[0]
            vol_param = params_vector[1]
            self.model.parameters['a'] = mean_rever_param
            self.model.parameters['sigma'] = vol_param
        else:
            vol_param = params_vector[0]
            self.model.parameters['sigma'] = vol_param

        squared_errors_sum = 0.0
        num_instruments = len(self.mar_data_df)

        for instrument_idx in range(num_instruments):
            instrument_row = self.mar_data_df.iloc[instrument_idx]
            market_price_obs = float(instrument_row['Price'])
            strike_rate = self.rate_to_decimal(instrument_row['Strike'])
            notional_amount = float(instrument_row['Notional'])

            if instrument_row['InstrumentType'] == 'Caplet':
                option_expiry_time = float(instrument_row['Expiry'])
                rate_maturity_time = float(instrument_row['Maturity'])
                model_price_calc = float(self.pricer.caplet(
                    T1=option_expiry_time,
                    T2=rate_maturity_time,
                    N=notional_amount,
                    K=strike_rate,
                    method='cf'
                ))
            elif instrument_row['InstrumentType'] == 'Swap':
                swap_dates = instrument_row['Dates']
                payer_flag = instrument_row.get('PayerFlag', True)
                model_price_calc = float(self.pricer.swap(
                    Tau=swap_dates,
                    N=notional_amount,
                    K=strike_rate,
                    payer=payer_flag,   
                    mc=False
                ))
            else:
                raise ValueError(f"Unknown instrument type: {instrument_row['InstrumentType']}")

            price_delta = model_price_calc - market_price_obs

            if self.weight_scheme == 'Relative':
                weight_factor = 1.0 / (abs(market_price_obs) + 1e-8)
            elif self.weight_scheme == 'Vega' and 'Vega' in instrument_row:
                weight_factor = 1.0 / (instrument_row['Vega'] + 1e-8)
            else:
                weight_factor = 1.0

            squared_errors_sum += (price_delta * weight_factor) ** 2

        rmse_error = np.sqrt(squared_errors_sum / max(num_instruments, 1))
        self.calibration_history.append((params_vector.copy() if isinstance(params_vector, np.ndarray) 
                                         else params_vector, rmse_error))
        
        return rmse_error


    def calibrate(self):
        curr_mean_reversion = self.model.parameters['a']
        initial_volatility = 0.01

        if self.calibrate_mean_reversion:
            params_initial = np.array([curr_mean_reversion, initial_volatility])
            bounds_constraints = [(1e-4, 1.0), (1e-5, 0.2)]
        else:
            params_initial = np.array([initial_volatility])
            bounds_constraints = [(1e-5, 0.2)]

        opt_result = minimize(
            self.calculate_objective_error,
            params_initial,
            method='L-BFGS-B',
            bounds=bounds_constraints
        )

        if opt_result.success:
            if self.calibrate_mean_reversion:
                opt_mean_rever = float(opt_result.x[0])
                opt_volatility = float(opt_result.x[1])
                self.model.parameters['a'] = opt_mean_rever
                self.model.parameters['sigma'] = opt_volatility
                print("\nCalibration Successful")
                print(f"\nOptimal Mean Reversion (a): {opt_mean_rever:.6f}")
                print(f"Optimal Volatility (sigma): {opt_volatility:.6f}")
            else:
                opt_volatility = float(opt_result.x[0])
                self.model.parameters['sigma'] = opt_volatility
                print("\nCalibration Successful")
                print(f"\nOptimal Volatility (sigma): {opt_volatility:.6f}")

            final_rmse = float(opt_result.fun)
            print(f"Final RMSE: {final_rmse:.6f}")
        else:
            print("Calibration Failed:", opt_result.message)

        return opt_result


def invert_bach_normal_vol(mar_price_input, frd_rate, strike_rate, time_to_expiry, 
                                 annuity_factor, not_size):
    frd_dec = frd_rate / 100.0 if frd_rate and frd_rate > 0.5 else frd_rate
    strike_dec = strike_rate / 100.0 if strike_rate and strike_rate > 0.5 else strike_rate

    if time_to_expiry <= 0 or annuity_factor <= 0 or not_size <= 0:
        raise ValueError("time_to_expiry, annuity_factor, and not_size must be positive")

    target_price = float(mar_price_input)

    def bach_formula_price(normal_vol):
        if normal_vol <= 0:
            return np.nan
        sqrt_time = np.sqrt(time_to_expiry)
        d_param = (frd_dec - strike_dec) / (normal_vol * sqrt_time)
        bach_price = annuity_factor * not_size * ((frd_dec - strike_dec) * norm.cdf(d_param) 
                                                   + normal_vol * sqrt_time * norm.pdf(d_param))

        return bach_price
    
    lower_vol_bound = 1e-8
    upper_vol_bound = 1.0

    try:
        price_at_lower = bach_formula_price(lower_vol_bound) - target_price
        price_at_upper = bach_formula_price(upper_vol_bound) - target_price
        
        if price_at_lower * price_at_upper > 0:
            for _ in range(6):
                upper_vol_bound *= 2.0
                price_at_upper = bach_formula_price(upper_vol_bound) - target_price
                if price_at_lower * price_at_upper <= 0:
                    break

        implied_normal_vol = brentq(
            lambda vol_guess: bach_formula_price(vol_guess) - target_price,
            lower_vol_bound,
            upper_vol_bound
        )
    except ValueError:
        return np.nan

    return implied_normal_vol * 10000.0

if __name__ == "__main__":
    print("Calibration module works")
