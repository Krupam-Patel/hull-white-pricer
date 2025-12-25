"""Calibrate Hull White model sigma with fixed a."""
"""
DONE
"""
from scipy.optimize import minimize, brentq
from scipy.stats import norm
import numpy as np


class hwCalibrator:
    def __init__(self, pricer, market_prices, calibrate_to="Caplets", a_fixed=0.01):
        self.pricer = pricer
        self.model = pricer.model

        self.market_prices = market_prices
        self.calibrate_to = calibrate_to    

        self.a_fixed = float(a_fixed)
        self.history = []
        self.model.parameters["a"] = self.a_fixed

    def maybe_rate(self, x):
        return x / 100.0 if x and x > 0.5 else x

    def objective(self, sigma_vec):
        sigma = float(sigma_vec[0])
        self.model.parameters["sigma"] = sigma

        prices = self.market_prices["Prices"]
        sq_err = 0.0

        for i in range(len(prices)):
            mkt_price = float(prices[i])
            K = self.maybe_rate(self.market_prices["Strike"][i])
            N = float(self.market_prices["Notional"][i])

            if self.calibrate_to == "Caplets":
                T = float(self.market_prices["Expiry"][i])
                S = float(self.market_prices["Maturity"][i])
                model_price = float(self.pricer.caplet(T, S, N, K))
            elif self.calibrate_to == "Swaptions":
                D = self.market_prices["Dates"][i]
                model_price = float(self.pricer.swaption(D, N, K))
            else:
                raise ValueError("calibrate_to must be Caplets or Swaptions")

            diff = model_price - mkt_price
            sq_err += diff * diff

        rmse = np.sqrt(sq_err / max(len(prices), 1))
        self.history.append((sigma, rmse))
        return rmse

    def callback(self, sigma_vec):
        sigma = float(sigma_vec[0])
        if self.history:
            _, last_rmse = self.history[-1]
            print(self.a_fixed, sigma, last_rmse)

    def calibrate(self):
        init_sigma = 0.01
        bounds = ((1e-5, 0.20),)
        method = "L-BFGS-B" 

        x0 = [float(init_sigma)]

        result = minimize(
            self.objective,
            x0,
            method="L-BFGS-B",
            bounds=bounds,
        )

        if result.success:
            opt_sigma = float(result.x[0])
            self.model.parameters["sigma"] = opt_sigma
            print("\nCalibration Successful")
            print(f"\nOptimal Sigma: {opt_sigma:.6f}")
            print(f"\nFinal RMSE: {float(result.fun):.6f}")
        else:
            print("Calibration Failed:", result.message)

        return result


def black_normal_vol(price, forward, strike, expiry, notional, annuity):
    f = forward / 100.0 if forward and forward > 0.5 else forward
    k = strike  / 100.0 if strike  and strike  > 0.5 else strike

    if expiry <= 0 or annuity <= 0 or notional <= 0:
        raise ValueError("expiry, annuity, and notional must be positive")

    target = float(price)

    def bach_price(sig):
        if sig <= 0:
            return np.nan
        root_t = np.sqrt(expiry)
        d = (f - k) / (sig * root_t)
        return annuity * notional * ((f - k) * norm.cdf(d) + sig * root_t * norm.pdf(d))

    lo = 1e-8
    hi = 1.0

    try:
        px_lo = bach_price(lo) - target
        px_hi = bach_price(hi) - target
        if px_lo * px_hi > 0:
            for _ in range(6):
                hi *= 2.0
                px_hi = bach_price(hi) - target
                if px_lo * px_hi <= 0:
                    break

        sigma_normal = brentq(lambda s: bach_price(s) - target, lo, hi)
    except ValueError:
        return np.nan

    return sigma_normal * 10000.0

if __name__ == "__main__":
    print("Calibration module works")
