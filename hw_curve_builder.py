"""Market curve for discount factors and forward rates."""
"""
DONE
"""
import numpy as np
from scipy.interpolate import PchipInterpolator

class hwCurve:
    def __init__(self, time, discount_factors):
        t = np.array(time)
        df = np.array(discount_factors)
        if t[0] != 0:
            t = np.insert(t, 0, 0.0)
            df = np.insert(df, 0, 1.0)
            
        self.log_df= PchipInterpolator(t, np.log(df))

    def discount(self, t):
        return np.exp(self.log_df(t))
    
    def inst_forward_rate(self, t):
        return -self.log_df.derivative(1)(t)
    
    def forward_rate_deriv(self, t):
        return -self.log_df.derivative(2)(t)
    
    def forward_rate(self, T1, T2):
        if np.any(np.abs(T2 - T1) < 1e-8):
            return self.inst_forward_rate(T1)
            
        p1 = self.discount(T1)
        p2 = self.discount(T2)
        return (p1 / p2 - 1.0) / (T2 - T1)

    def zero_rate(self, t):
        if np.any(t == 0):
            return self.inst_forward_rate(0)
        return -np.log(self.discount(t)) / t
    

if __name__ == "__main__":
    print("Curve module works.")
