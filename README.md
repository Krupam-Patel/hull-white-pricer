# Hull-White Interest Rate Model Library

This repository provides a set of Python modules for building discount curves, implementing the Hull-White short-rate model, simulating interest-rate paths, pricing interest-rate derivatives, and calibrating the model to market data. It is intended for quantitative finance, and researchers focused on interest rate modeling and derivatives valuation.

---

## Modules

### 1. `curve_builder.py`

- Implements the `Curve` class for managing market discount curves
- Uses cubic interpolation and smoothing splines to construct discount factors and instantaneous forward rates
- Provides methods to obtain discount factors and forward rates for arbitrary maturities

### 2. `hw_calibration.py`
- Implements the `HullWhiteCalibrator` class to calibrate the Hull-White model parameters (`a` and `sigma`) to market prices of caps or other interest rate instruments
- Uses nonlinear least-squares optimization (`scipy.optimize.minimize`) to minimize pricing errors

### 3. `hw_model.py`
- Implements the Hull-White one-factor short rate model
- `HullWhiteModel`: encapsulates the Hull-White model dynamics and analytic functions
- `HullWhiteSimulation`: Monte Carlo simulator for short rate paths using exact and Euler schemes under both the risk-neutral and forward measures
- `HullWhiteCurveBuilder`: computes zero-coupon bond prices, discount factors, forward rates, and long rates implied by Hull-White dynamics
  
### 4. `hw_pricer.py`
- Implements pricing engines for interest rate derivatives
- `HullWhitePricer`: prices standard interest rate products including zero-coupon bond options (calls and puts), caps, floors, swaps, and swaptions using both analytic formulas and Monte Carlo methods

---

## Features

- Smooth interpolation of market discount curves
- Analytical and simulation-based methods for interest rate dynamics
- Pricing of a broad range of interest rate derivatives
- Model implementation under the forward measure for efficient computations
- Calibration framework suitable for real market data

---

## Sources I used to make this algorithm 

- Perplexity / ChatGPT
- A lottttt of Youtube (Especially the goat Nicholas Burgess (https://www.youtube.com/@AlgoQuantHub)
    - The absolute goat and really suggest you watch him 
- https://www.tutorialspoint.com/scipy/scipy_interpolate_pchipinterpolator_function.htm
- https://docs.python.org/3/howto/logging.html
- https://www.wallstreetprep.com/knowledge/discount-factor/

### Research Papers
- https://www.casact.org/sites/default/files/old/oncourses_module4_ahlgrim.pdf
- https://www.math.kth.se/matstat/seminarier/reports/M-exjobb12/120220b.pdf (Arnaud Blanchard)
- https://arxiv.org/pdf/1808.03463 (Julian Holzermann)
