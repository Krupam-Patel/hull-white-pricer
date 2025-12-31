"""Market curve for discount factors and forward rates"""
#Fully done 

import numpy as np
from scipy.interpolate import PchipInterpolator

class hwCurve:
    def __init__(self, time_points, discount_factors):
        time_points = np.array(time_points, dtype=float)
        discount_factors = np.array(discount_factors, dtype=float)

        if time_points[0] != 0.0:
            time_points = np.insert(time_points, 0, 0.0)
            discount_factors = np.insert(discount_factors, 0, 1.0)

        self.times = time_points
        self.dfs = discount_factors
        self.log_df_interp = PchipInterpolator(time_points, np.log(discount_factors))

    def disc(self, target_time):
        log_df = self.log_df_interp(target_time)
        return np.exp(log_df)

    def inst_fwd_rate(self, maturity):
        return -self.log_df_interp.derivative(1)(maturity)

    def fwd_rate_deriv(self, maturity):
        return -self.log_df_interp.derivative(2)(maturity)
    
    def fwd_rate(self, start_time, end_time):
        if np.any(np.abs(end_time - start_time) < 1e-8):
            return self.inst_fwd_rate(start_time)

        p_start = self.disc(start_time)
        p_end = self.disc(end_time)
        return (p_start / p_end - 1.0) / (end_time - start_time)

    def zero_rate(self, maturity):
        if np.any(maturity == 0):
            return self.inst_fwd_rate(0)
        return -np.log(self.disc(maturity)) / maturity

    def par_rate(self, start_time, end_time, payment_freq=0.5):
        num_payments = int(round((end_time - start_time) / payment_freq))
        if num_payments == 0:
            return 0.0

        payment_times = np.linspace(start_time, end_time, num_payments + 1)[1:]
        annuity = np.sum(self.disc(payment_times) * payment_freq)
        pv_float = self.disc(start_time) - self.disc(end_time)

        return pv_float / annuity if annuity > 1e-12 else 0.0

    def check_montonicity(self, tol=1e-10):
        if np.any(np.diff(self.dfs) > tol):
            raise ValueError("Discount factors must be non-increasing (non-negative forward rates).")

    def repr(self):
        return f"hwCurve(times={self.times}, dfs={self.dfs})"


if __name__ == "__main__":
    print("Curve module works")
