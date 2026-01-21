import logging
from typing import Optional, Sequence, Tuple, Union

#Fully done 1-21-2026

import numpy as np
from scipy.optimize import differential_evolution, minimize, brentq
from scipy.stats import norm

logger = logging.getLogger(__name__)

def rate_to_decimal(rate_input: Optional[float], percent_threshold: float = 0.5) -> Optional[float]:
    if rate_input is None:
        return None
    return rate_input / 100.0 if rate_input > percent_threshold else rate_input

class hwCalibrator:
    def __init__(self, model, pricer, mar_data_df, weights, mean_rever: bool = True, verbose: bool = True):
        self.model = model
        self.pricer = pricer
        self.mar_data_df = mar_data_df
        self.weights = np.asarray(weights, dtype=float)
        self.mean_rever = mean_rever
        self.verbose = verbose
        self.cali_hist: list[Tuple[np.ndarray, float]] = []

    def log(self, msg: str) -> None:
        if self.verbose:
            logger.info(msg)

    def objective_error(self, params_vector: Sequence[float]) -> float:
        if self.mean_rever:
            mean_rever_param = float(params_vector[0])
            vol_param = float(params_vector[1])
            self.model.parameters["a"] = mean_rever_param
            self.model.parameters["sigma"] = vol_param
        else:
            vol_param = float(params_vector[0])
            self.model.parameters["sigma"] = vol_param

        num_instruments = len(self.mar_data_df)
        if num_instruments == 0:
            return float(np.inf)

        squared_errors_sum = 0.0

        for instrument_idx in range(num_instruments):
            row = self.mar_data_df.iloc[instrument_idx]
            market_price = float(row["Price"])
            strike = rate_to_decimal(row["Strike"])
            notional = float(row["Notional"])
            w = float(self.weights[instrument_idx])

            inst_type = str(row["InstrumentType"])

            if inst_type == "Caplet":
                T1 = float(row["Expiry"])
                T2 = float(row["Maturity"])
                model_price = float(self.pricer.caplet(T1=T1, T2=T2, N=notional, K=strike, method="cf"))
            elif inst_type == "Swap":
                dates = row["Dates"]
                payer_flag = row.get("PayerFlag", True)
                model_price = float(self.pricer.swap(Tau=dates, N=notional, K=strike, payer=payer_flag, mc=False))
            else:
                raise ValueError(f"Unknown instrument type: {inst_type}")

            diff = model_price - market_price
            squared_errors_sum += (diff * w) ** 2

        rmse = float(np.sqrt(squared_errors_sum / num_instruments))
        params_copy = np.array(params_vector, dtype=float).copy()
        self.cali_hist.append((params_copy, rmse))

        return rmse

    def calibrate(self,use_global: bool = True,de_tol: float = 1e-4,de_maxiter: int = 200, lbfgs_options: Optional[dict] = None):
        if lbfgs_options is None:
            lbfgs_options = {"ftol": 1e-9, "gtol": 1e-6, "maxiter": 500}

        curr_a = float(self.model.parameters.get("a", 0.05))
        initial_sigma = 0.01

        if self.mean_rever:
            bounds = [(1e-4, 1.0), (1e-5, 0.2)]
            x0 = np.array([curr_a, initial_sigma], dtype=float)
        else:
            bounds = [(1e-5, 0.2)]
            x0 = np.array([initial_sigma], dtype=float)

        if use_global:
            result = differential_evolution(self.objective_error, bounds, strategy="best1bin", polish=True, tol=de_tol, maxiter=de_maxiter)
        else:
            result = minimize(self.objective_error, x0, method="L-BFGS-B", bounds=bounds, options=lbfgs_options)

        if result.success:
            opt_params = np.array(result.x, dtype=float)
            if self.mean_rever:
                opt_a = float(opt_params[0])
                opt_sigma = float(opt_params[1])
                self.model.parameters["a"] = opt_a
                self.model.parameters["sigma"] = opt_sigma
                self.log("Calibration Successful")
                self.log(f"Optimal Mean Reversion (a): {opt_a:.6f}")
                self.log(f"Optimal Volatility (sigma): {opt_sigma:.6f}")
            else:
                opt_sigma = float(opt_params[0])
                self.model.parameters["sigma"] = opt_sigma
                self.log("Calibration Successful")
                self.log(f"Optimal Volatility (sigma): {opt_sigma:.6f}")

            final_rmse = float(result.fun)
            self.log(f"Final RMSE: {final_rmse:.6f}")
        else:
            self.log(f"Calibration Failed: {result.message}")

        return result


def invert_bach_normal_vol(
    mar_price_input: float,
    frd_rate: float,
    k_rate: float,
    time_to_expiry: float,
    annuity_factor: float,
    notional_size: float,
    percent_threshold: float = 0.5,
    max_expand: int = 10,
    return_in_bps: bool = True) -> float:

    frd_dec = rate_to_decimal(frd_rate, percent_threshold=percent_threshold)
    k_dec = rate_to_decimal(k_rate, percent_threshold=percent_threshold)

    if frd_dec is None or k_dec is None:
        raise ValueError("frd_rate and k_rate must be non-None")

    if time_to_expiry <= 0 or annuity_factor <= 0 or notional_size <= 0:
        raise ValueError("time_to_expiry, annuity_factor, and notional_size must be positive")

    target_price = float(mar_price_input)

    def bach_formula_price(normal_vol: float) -> float:
        if normal_vol <= 0:
            return np.nan
        sqrt_t = np.sqrt(time_to_expiry)
        d = (frd_dec - k_dec) / (normal_vol * sqrt_t)
        price = (annuity_factor * notional_size * ((frd_dec - k_dec) * norm.cdf(d) + normal_vol * sqrt_t * norm.pdf(d)))

        return float(price)

    lower_vol = 1e-8
    upper_vol = 1.0

    try:
        f_low = bach_formula_price(lower_vol) - target_price
        f_up = bach_formula_price(upper_vol) - target_price

        if np.isnan(f_low) or np.isnan(f_up):
            return float(np.nan)

        if f_low * f_up > 0:
            for _ in range(max_expand):
                upper_vol *= 2.0
                f_up = bach_formula_price(upper_vol) - target_price
                if np.isnan(f_up):
                    return float(np.nan)
                if f_low * f_up <= 0:
                    break

        implied_vol = brentq(
            lambda v: bach_formula_price(v) - target_price,
            lower_vol,
            upper_vol,
        )
    except ValueError:
        return float(np.nan)

    if return_in_bps:
        return float(implied_vol * 10_000.0)
    return float(implied_vol)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    logger.info("Calibration module imported and ready.")

