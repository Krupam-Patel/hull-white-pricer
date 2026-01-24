"""Market curve for discount factors and forward rates"""
# Fully done 1-20-2026
# Comments done 1-24-2026

# I tried putting comments to explain my thought process for everything in the code
# If something is still fuzzy, there is a typo, or there is a better way to do something, please dm on linkedin!
# https://www.linkedin.com/in/krupam-patel/


import numpy as np
from scipy.interpolate import PchipInterpolator as Pchip


class hwCurve:
    def __init__(self, time_points, disc_factors):
        # Stores curve times and discount factors as clean NumPy arrays
        time_points = np.array(time_points, dtype=float)
        disc_factors = np.array(disc_factors, dtype=float)

        # Ensure the curve is anchored at time 0 with discount factor = 1
        if time_points[0] != 0.0:
            time_points = np.insert(time_points, 0, 0.0)
            disc_factors = np.insert(disc_factors, 0, 1.0)

        self.times = time_points
        self.dfs = disc_factors

        # Interpolator working in log‑discount‑factor space
        # If you want to learn about this, use https://www.tutorialspoint.com/scipy/scipy_interpolate_pchipinterpolator_function.htm
        self.log_df_interp = Pchip(time_points, np.log(disc_factors))

    def discount(self, target_time: float) -> float | np.ndarray:
        # Discount factor at a given time using the interpolated curve
        log_df = self.log_df_interp(target_time)
        return np.exp(log_df)

    def inst_fwd_rate(self, maturity: float) -> float | np.ndarray:
        # Instantaneous forward rate at a given maturity
        return -self.log_df_interp.derivative(1)(maturity)

    def fwd_rate_deriv(self, maturity: float) -> float | np.ndarray:
        # Derivative of the instantaneous forward rate
        return -self.log_df_interp.derivative(2)(maturity)

    def fwd_rate(self, start_time: float, end_time: float) -> float | np.ndarray:
        # Simple forward rate between two times from discount factors
        if np.any(np.abs(end_time - start_time) < 1e-8):
            return self.inst_fwd_rate(start_time)

        df_start = self.discount(start_time)
        df_end = self.discount(end_time)

        return (df_start / df_end - 1.0) / (end_time - start_time)

    def zero_rate(self, maturity: float) -> float | np.ndarray:
        # Continuously‑compounded zero rate up to a given maturity
        if np.any(maturity == 0.0):
            return self.inst_fwd_rate(0.0)
        return -np.log(self.discount(maturity)) / maturity

    def par_rate(self, start_time: float, end_time: float, payment_freq: float = 0.5) -> float:
        # Par coupon rate for a swap/bond between start_time and end_time
        num_payments = int(round((end_time - start_time) / payment_freq))
        if num_payments <= 0:
            return 0.0

        payment_times = start_time + payment_freq * np.arange(1, num_payments + 1)

        disc_start = self.discount(start_time)
        disc_end = self.discount(end_time)
        disc_flows = self.discount(payment_times)

        # PV of 1 unit of coupon paid each period
        annuity_factor = np.sum(payment_freq * disc_flows)

        if annuity_factor == 0.0:
            return 0.0

        return (disc_start - disc_end) / annuity_factor

    def check_monotonicity(self, tol: float = 1e-10) -> None:
        # Makes sure discount factor should not increase with maturity
        if np.any(np.diff(self.dfs) > tol):
            raise ValueError("Discount factors must be non-increasing.")

    def __repr__(self) -> str:
        # Compact string representation of the curve
        return f"hwCurve(times={self.times}, dfs={self.dfs})"


if __name__ == "__main__":
    # Builds a curve and print a sample par rate
    curve = hwCurve([1.0, 2.0], [0.95, 0.90])
    print(f"Par rate (1Y to 2Y): {curve.par_rate(1.0, 2.0):.4%}")
