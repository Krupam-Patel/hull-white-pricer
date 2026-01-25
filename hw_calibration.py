"""Hull–White calibration and Bachelier implied volatility utilities"""
# Fully done 1/20/2026
# Comments fully done 1/25/2026

# I tried putting comments to explain my thought process for everything in the code
# If something is still fuzzy, there is a typo, or there is a better way to do something, please dm on linkedin!
# https://www.linkedin.com/in/krupam-patel/

import logging
from typing import Optional, Sequence, Tuple
import numpy as np
from scipy.optimize import differential_evolution, minimize, brentq
from scipy.stats import norm

logger = logging.getLogger(__name__)

def convert_rate_to_decimal(rate: Optional[float], percent_threshold: float = 0.5) -> Optional[float]:
    # Converts percentage rates (e.g., 5.0%) to decimals (e.g., 0.05)
    if rate is None:
        return None
    return rate / 100.0 if rate > percent_threshold else rate

class HullWhiteCalibrator:
    def __init__(self, model, pricer, market_data_df, instrument_weights, calibrate_mean_reversion: bool = True, verbose: bool = True):
        # Initializes with model, pricer, market data DataFrame, and calibration options
        self.model = model
        self.pricer = pricer
        self.market_data_df = market_data_df
        self.instrument_weights = np.asarray(instrument_weights, dtype=float)
        self.calibrate_mean_reversion = calibrate_mean_reversion
        self.verbose = verbose
        self.calibration_history: list[Tuple[np.ndarray, float]] = []

    def log_message(self, message: str) -> None:
        # Logs message if verbose mode is enabled
        if self.verbose:
            logger.info(message)

    def rmse_objective(self, parameters: Sequence[float]) -> float:
        # Computes RMSE between model and market prices across all instruments
        if self.calibrate_mean_reversion:
            mean_reversion = float(parameters[0])
            volatility = float(parameters[1])
            self.model.parameters["a"] = mean_reversion
            self.model.parameters["sigma"] = volatility
        else:
            volatility = float(parameters[0])
            self.model.parameters["sigma"] = volatility

        num_instruments = len(self.market_data_df)
        if num_instruments == 0:
            return float("inf")

        total_squared_error = 0.0

        for idx in range(num_instruments):
            row = self.market_data_df.iloc[idx]
            mrkt_price = float(row["Price"])
            k = convert_rate_to_decimal(row["Strike"])
            notional = float(row["Notional"])
            weight = float(self.instrument_weights[idx])

            instrument_type = str(row["InstrumentType"])

            if instrument_type == "Caplet":
                expiry = float(row["Expiry"])
                maturity = float(row["Maturity"])
                model_price = float(self.pricer.caplet(T1=expiry, T2=maturity, N=notional, K=k, method="cf"))
            elif instrument_type == "Swap":
                payment_dates = row["Dates"]
                is_payer = row.get("PayerFlag", True)
                model_price = float(self.pricer.swap(Tau=payment_dates, N=notional, K=k, payer=is_payer, mc=False))
            else:
                raise ValueError(f"Unsupported instrument: {instrument_type}")

            price_error = model_price - mrkt_price
            total_squared_error += (price_error * weight) ** 2

        rmse = np.sqrt(total_squared_error / num_instruments)
        param_copy = np.array(parameters, dtype=float).copy()
        self.calibration_history.append((param_copy, float(rmse)))

        return float(rmse)

    def run_calibration(self, use_global_search: bool = True, global_tol: float = 1e-4, global_maxiter: int = 200, local_options: Optional[dict] = None):
        # Performs global (differential evolution) or local (L-BFGS-B) parameter optimization
        if local_options is None:
            local_options = {"ftol": 1e-9, "gtol": 1e-6, "maxiter": 500}

        current_mean_reversion = float(self.model.parameters.get("a", 0.05))
        initial_volatility = 0.01

        if self.calibrate_mean_reversion:
            parameter_bounds = [(1e-4, 1.0), (1e-5, 0.2)]
            initial_guess = np.array([current_mean_reversion, initial_volatility], dtype=float)
        else:
            parameter_bounds = [(1e-5, 0.2)]
            initial_guess = np.array([initial_volatility], dtype=float)

        if use_global_search:
            optimization_result = differential_evolution(self.rmse_objective, parameter_bounds, strategy="best1bin", polish=True, tol=global_tol, maxiter=global_maxiter)
        else:
            optimization_result = minimize(self.rmse_objective, initial_guess, method="L-BFGS-B", bounds=parameter_bounds, options=local_options)

        if optimization_result.success:
            optimal_params = np.array(optimization_result.x, dtype=float)
            if self.calibrate_mean_reversion:
                optimal_mean_reversion = float(optimal_params[0])
                optimal_volatility = float(optimal_params[1])
                self.model.parameters["a"] = optimal_mean_reversion
                self.model.parameters["sigma"] = optimal_volatility
                self.log_message("Calibration completed successfully")
                self.log_message(f"Optimal mean reversion (a): {optimal_mean_reversion:.6f}")
                self.log_message(f"Optimal volatility (sigma): {optimal_volatility:.6f}")
            else:
                optimal_volatility = float(optimal_params[0])
                self.model.parameters["sigma"] = optimal_volatility
                self.log_message("Calibration completed successfully")
                self.log_message(f"Optimal volatility (sigma): {optimal_volatility:.6f}")

            final_rmse = float(optimization_result.fun)
            self.log_message(f"Achieved final RMSE: {final_rmse:.6f}")
        else:
            self.log_message(f"Calibration failed: {optimization_result.message}")

        return optimization_result

def invert_bachelier_volatility(mrkt_price: float, fwd_rate: float, k_rate: float, time_to_expiry: float, annuity: float, 
                                notional: float, percent_threshold: float = 0.5, max_bound_expansions: int = 10, return_in_bps: bool = True) -> float:

    # Computes implied normal volatility from cap/floor price using Bachelier model
    fwd_decimal = convert_rate_to_decimal(fwd_rate, percent_threshold=percent_threshold)
    k_decimal = convert_rate_to_decimal(k_rate, percent_threshold=percent_threshold)

    if fwd_decimal is None or k_decimal is None:
        raise ValueError("Fwd rate and k must be valid numbers")

    if time_to_expiry <= 0 or annuity <= 0 or notional <= 0:
        raise ValueError("Time to expiry, annuity, and notional must be positive")

    target_price = float(mrkt_price)

    def bachelier_price(normal_vol: float) -> float:
        # Calculates caplet/floorlet price under Bachelier (normal) model
        if normal_vol <= 0:
            return np.nan
        time_sqrt = np.sqrt(time_to_expiry)
        d = (fwd_decimal - k_decimal) / (normal_vol * time_sqrt)
        price = annuity * notional * ((fwd_decimal - k_decimal) * norm.cdf(d) + normal_vol * time_sqrt * norm.pdf(d))

        return float(price)

    low_vol = 1e-8
    high_vol = 1.0

    try:
        low_price_diff = bachelier_price(low_vol) - target_price
        high_price_diff = bachelier_price(high_vol) - target_price

        if np.isnan(low_price_diff) or np.isnan(high_price_diff):
            return float(np.nan)

        # Expand upper bound if root not bracketed
        if low_price_diff * high_price_diff > 0:
            for _ in range(max_bound_expansions):
                high_vol *= 2.0
                high_price_diff = bachelier_price(high_vol) - target_price
                if np.isnan(high_price_diff):
                    return float(np.nan)
                if low_price_diff * high_price_diff <= 0:
                    break

        implied_vol = brentq(lambda v: bachelier_price(v) - target_price, low_vol, high_vol)
    except ValueError:
        return float(np.nan)

    return float(implied_vol * 10000.0) if return_in_bps else float(implied_vol)

if __name__ == "__main__":
    # Sets up logging for direct module execution
    logging.basicConfig(level=logging.INFO)
    logger.info("Hull-White calibration module ready")

