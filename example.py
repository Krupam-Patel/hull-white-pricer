from hw_pricer import hwPricer

import numpy as np
import pandas as pd
from numpy.typing import NDArray


def example_basic_pricing(pricer: 'hwPricer') -> None:
    print("=" * 70)
    print("BASIC PRICING EXAMPLES")
    print("=" * 70)
    option_expiry_years: float = 1.0
    bond_maturity_years: float = 3.0
    strike_price: float = 0.97
    
    bond_put_price: float = pricer.zero_bond_put(
        option_expiry=option_expiry_years,
        bond_maturity=bond_maturity_years,
        strike=strike_price,
        mc=False
    )
    
    bond_call_price: float = pricer.zero_bond_call(
        option_expiry=option_expiry_years,
        bond_maturity=bond_maturity_years,
        strike=strike_price,
        mc=False
    )
    
    print(f"\nZero-Coupon Bond Options (Analytical):")
    print(f"  Expiry: {option_expiry_years:.2f} years, Maturity: {bond_maturity_years:.2f} years")
    print(f"  Put Price (Strike={strike_price}): ${bond_put_price:.6f}")
    print(f"  Call Price (Strike={strike_price}): ${bond_call_price:.6f}")
    fixing_date_years: float = 0.5
    payment_date_years: float = 1.0
    notional_amount: float = 1_000_000.0
    cap_strike_rate: float = 0.03
    
    caplet_price_analytical: float = pricer.caplet(
        fixing_time=fixing_date_years,
        payment_time=payment_date_years,
        notional=notional_amount,
        strike=cap_strike_rate,
        method='cf'
    )
    
    caplet_price_jamshidian: float = pricer.caplet(
        fixing_time=fixing_date_years,
        payment_time=payment_date_years,
        notional=notional_amount,
        strike=cap_strike_rate,
        method='js'
    )
    
    print(f"\nCaplet Pricing (Notional=${notional_amount/1e6:.0f}M, Strike={cap_strike_rate*100:.1f}%):")
    print(f"  Closed-Form Price: ${caplet_price_analytical:.2f}")
    print(f"  Jamshidian Price: ${caplet_price_jamshidian:.2f}")


def example_cap_floor_pricing(pricer: 'hwPricer') -> None:
    print("\n" + "=" * 70)
    print("CAP AND FLOOR EXAMPLES")
    print("=" * 70)
    payment_dates: NDArray[np.floating] = np.array([
        0.25, 0.50, 0.75, 1.0, 1.25, 1.5, 1.75, 2.0
    ])
    
    notional_amount: float = 10_000_000.0
    cap_floor_strike: float = 0.025  # 2.5%
    cap_price: float = pricer.cap(
        payment_schedule=payment_dates,
        notional=notional_amount,
        strike=cap_floor_strike,
        mc=False
    )
    
    floor_price: float = pricer.floor(
        payment_schedule=payment_dates,
        notional=notional_amount,
        strike=cap_floor_strike,
        mc=False
    )
    
    print(f"\nCap and Floor Pricing:")
    print(f"  Notional: ${notional_amount/1e6:.0f}M")
    print(f"  Strike Rate: {cap_floor_strike*100:.2f}%")
    print(f"  Payment Schedule: {len(payment_dates)-1} periods (quarterly)")
    print(f"  Cap Price: ${cap_price:,.2f}")
    print(f"  Floor Price: ${floor_price:,.2f}")
    swap_intrinsic: float = pricer.swap(
        payment_schedule=payment_dates,
        notional=notional_amount,
        fixed_rate=cap_floor_strike,
        payer=True,
        mc=False
    )
    
    print(f"  Cap - Floor ≈ Swap Value: ${cap_price - floor_price:,.2f} vs ${swap_intrinsic:,.2f}")


def example_swaption_pricing(pricer: 'hwPricer') -> None:
    print("\n" + "=" * 70)
    print("SWAPTION EXAMPLES")
    print("=" * 70)
    swaption_expiry_years: float = 1.0
    swap_payment_dates: NDArray[np.floating] = np.array([
        swaption_expiry_years,
        swaption_expiry_years + 0.25,
        swaption_expiry_years + 0.50,
        swaption_expiry_years + 0.75,
        swaption_expiry_years + 1.0,
    ])
    
    notional_amount: float = 50_000_000.0
    underlying_swap_rate: float = 0.027  # 2.7%
    payer_swaption_price: float = pricer.swaption(
        payment_schedule=swap_payment_dates,
        notional=notional_amount,
        fixed_rate=underlying_swap_rate,
        payer=True,
        mc=False
    )
    
    receiver_swaption_price: float = pricer.swaption(
        payment_schedule=swap_payment_dates,
        notional=notional_amount,
        fixed_rate=underlying_swap_rate,
        payer=False,
        mc=False
    )
    
    print(f"\n{swaption_expiry_years:.1f}Y x {swap_payment_dates[-1] - swaption_expiry_years:.1f}Y Swaption:")
    print(f"  Notional: ${notional_amount/1e6:.0f}M")
    print(f"  Fixed Rate: {underlying_swap_rate*100:.2f}%")
    print(f"  Payer Swaption Price: ${payer_swaption_price:,.2f}")
    print(f"  Receiver Swaption Price: ${receiver_swaption_price:,.2f}")


def example_bond_option_pricing(pricer: 'hwPricer') -> None:
    print("\n" + "=" * 70)
    print("BOND OPTION EXAMPLES")
    print("=" * 70)
    option_expiry_years: float = 1.0
    coupon_payment_dates: NDArray[np.floating] = np.array([
        1.0, 1.25, 1.50, 1.75, 2.0, 2.25, 2.50, 2.75, 3.0
    ])
    
    coupon_rate: float = 0.04  # 4% coupon
    bond_notional: float = 100.0
    call_strike_price: float = 101.0
    bond_call_price: float = pricer.bond_option(
        option_expiry=option_expiry_years,
        payment_schedule=coupon_payment_dates,
        coupon_rate=coupon_rate,
        strike=call_strike_price,
        notional=bond_notional,
        call=True,
        mc=False
    )
    
    put_strike_price: float = 99.0
    bond_put_price: float = pricer.bond_option(
        option_expiry=option_expiry_years,
        payment_schedule=coupon_payment_dates,
        coupon_rate=coupon_rate,
        strike=put_strike_price,
        notional=bond_notional,
        call=False,
        mc=False
    )
    
    straight_bond_price: float = pricer.coupon_bond(
        payment_schedule=coupon_payment_dates,
        coupon_rate=coupon_rate,
        notional=bond_notional
    )
    
    print(f"\nBond Options ({coupon_rate*100:.1f}% Coupon Bond):")
    print(f"  Bond Par: ${bond_notional:.2f}")
    print(f"  Straight Bond Price: ${straight_bond_price:.2f}")
    print(f"  Call Price (Strike=${call_strike_price:.2f}): ${bond_call_price:.2f}")
    print(f"  Put Price (Strike=${put_strike_price:.2f}): ${bond_put_price:.2f}")
    callable_bond_price: float = straight_bond_price - bond_call_price
    print(f"  Callable Bond Price ≈ {straight_bond_price:.2f} - {bond_call_price:.2f} = ${callable_bond_price:.2f}")


def example_monte_carlo_pricing(pricer: 'hwPricer') -> None:
    print("\n" + "=" * 70)
    print("MONTE CARLO VS ANALYTICAL COMPARISON")
    print("=" * 70)
    option_expiry_years: float = 1.0
    bond_maturity_years: float = 2.5
    strike_price: float = 0.95
    
    put_price_analytical: float = pricer.zero_bond_put(
        option_expiry=option_expiry_years,
        bond_maturity=bond_maturity_years,
        strike=strike_price,
        mc=False
    )
    
    pricer.set_simulation(n_paths=50_000, seed=42)
    put_price_mc: float = pricer.zero_bond_put(
        option_expiry=option_expiry_years,
        bond_maturity=bond_maturity_years,
        strike=strike_price,
        mc=True
    )
    
    difference_bps: float = (put_price_mc - put_price_analytical) * 10_000
    
    print(f"\nZero-Coupon Bond Put Option:")
    print(f"  Expiry: {option_expiry_years:.2f} years, Maturity: {bond_maturity_years:.2f} years")
    print(f"  Analytical Price: ${put_price_analytical:.6f}")
    print(f"  Monte Carlo Price (50k paths): ${put_price_mc:.6f}")
    print(f"  Difference: {difference_bps:.2f} basis points")


def example_calibration_workflow(pricer: 'hwPricer', market_data: pd.DataFrame) -> None:
    print("\n" + "=" * 70)
    print("CALIBRATION WORKFLOW")
    print("=" * 70)
    
    from hw_calibration import hwCalibrator
    
    calibrator: hwCalibrator = hwCalibrator(
        pricer=pricer,
        mar_data_df=market_data,
        weight_scheme='Relative',
        mean_rever=False
    )
    
    print(f"\nMarket Data Summary:")
    print(f"  Instruments: {len(market_data)}")
    print(f"  Instrument Types: {market_data['InstrumentType'].unique()}")
    print(f"  Date Range: {market_data.index.min()} to {market_data.index.max()}")
    
    opt_result = calibrator.calibrate()
    
    if opt_result.success:
        print(f"\nCalibration successful!")
        print(f"  Final RMSE: {opt_result.fun:.6f}")
    else:
        print(f"Calibration failed: {opt_result.message}")


def example_greeks_and_sensitivities(pricer: 'hwPricer') -> None:
    print("\n" + "=" * 70)
    print("GREEKS AND SENSITIVITIES")
    print("=" * 70)
    option_expiry_years: float = 1.0
    bond_maturity_years: float = 2.5
    strike_price: float = 0.95
    
    base_price: float = pricer.zero_bond_put(
        option_expiry=option_expiry_years,
        bond_maturity=bond_maturity_years,
        strike=strike_price,
        mc=False
    )
    
    strike_bump: float = 0.001
    bumped_price: float = pricer.zero_bond_put(
        option_expiry=option_expiry_years,
        bond_maturity=bond_maturity_years,
        strike=strike_price + strike_bump,
        mc=False
    )
    delta: float = (bumped_price - base_price) / strike_bump
    
    bumped_price_down: float = pricer.zero_bond_put(
        option_expiry=option_expiry_years,
        bond_maturity=bond_maturity_years,
        strike=strike_price - strike_bump,
        mc=False
    )
    gamma: float = (bumped_price + bumped_price_down - 2 * base_price) / (strike_bump ** 2)
    
    print(f"\nZero-Coupon Bond Put Option Greeks:")
    print(f"  Base Price: ${base_price:.6f}")
    print(f"  Delta (per $1 strike move): ${delta:.6f}")
    print(f"  Gamma (per $1 strike move): ${gamma:.6f}")


if __name__ == "__main__":
    print("Hull-White Pricing Examples")
