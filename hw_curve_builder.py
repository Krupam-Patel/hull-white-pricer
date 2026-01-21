"""Market curve for discount factors and forward rates"""
# Fully done 1-20-2026

import numpy as np
from scipy.interpolate import PchipInterpolator as Pchip

class hwCurve:
    def __init__(self, time_points, disc_factors):
        time_points = np.array(time_points, dtype=float)
        disc_factors = np.array(disc_factors, dtype=float)

        if time_points[0] != 0.0:
            time_points = np.insert(time_points, 0, 0.0)
            disc_factors = np.insert(disc_factors, 0, 1.0)
            
        self.times = time_points
        self.dfs = disc_factors
        self.log_df_interp = Pchip(time_points, np.log(disc_factors))

    def discount(self, target_time: float) -> float | np.ndarray:
        log_df = self.log_df_interp(target_time)
        return np.exp(log_df)
        
    def inst_fwd_rate(self, maturity: float) -> float | np.ndarray:
        return -self.log_df_interp.derivative(1)(maturity)

    def fwd_rate_deriv(self, maturity: float) -> float | np.ndarray:
        return -self.log_df_interp.derivative(2)(maturity)

    def fwd_rate(self, start_time: float, end_time: float) -> float | np.ndarray:
        if np.any(np.abs(end_time - start_time) < 1e-8):
            return self.inst_fwd_rate(start_time)

        df_start = self.discount(start_time)
        df_end = self.discount(end_time)

        return (df_start / df_end - 1.0) / (end_time - start_time)

    def zero_rate(self, maturity: float) -> float | np.ndarray:
        if np.any(maturity == 0.0):
            return self.inst_fwd_rate(0.0)
        return -np.log(self.discount(maturity)) / maturity
    
    def par_rate(self, start_time: float, end_time: float, payment_freq: float = 0.5) -> float:
        num_payments = int(round((end_time - start_time) / payment_freq))
        if num_payments <= 0:
            return 0.0

        payment_times = start_time + payment_freq * np.arange(1, num_payments + 1)
        
        disc_start = self.discount(start_time)
        disc_end = self.discount(end_time)
        disc_flows = self.discount(payment_times)

        annuity_factor = np.sum(payment_freq * disc_flows)

        if annuity_factor == 0.0:
            return 0.0

        return (disc_start - disc_end) / annuity_factor
                 
    def check_monotonicity(self, tol=1e-10):
        if np.any(np.diff(self.dfs) > tol):
            raise ValueError("Discount factors must be non-increasing.")

    def __repr__(self):
        return f"hwCurve(times={self.times}, dfs={self.dfs})"

if __name__ == "__main__":
    curve = hwCurve([1.0, 2.0], [0.95, 0.90])
    print(f"Par rate (1Y to 2Y): {curve.par_rate(1.0, 2.0):.4%}")
