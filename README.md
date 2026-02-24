# ATR-Based Stop Loss Optimization for Small Caps

## Overview

This project implements a quantitative framework to determine optimal ATR-based Stop Loss (SL) multipliers under a Buying Power constraint for small-cap stocks.

The script analyzes historical OHLC data and:

- Extracts empirical SL coefficients
- Selects the coefficient that maintains average Buying Power within a defined range
- Simulates multiple TP/SL ratios
- Computes average trade duration

The focus is on volatility normalization (ATR) and capital efficiency through risk-based position sizing.

---

# How the Model Works

The process is divided into three main phases.

---

## 1️⃣ ATR Calculation

The script computes the Average True Range (ATR) over a rolling window:

```python
ATR = rolling_mean(True_Range, period=50)
```

True Range is defined as:

```python
max(
    High - Low,
    abs(High - Previous Close),
    abs(Low - Previous Close)
)
```

ATR is used to normalize Stop Loss distances according to market volatility.

---

## 2️⃣ Phase 1 — Empirical SL Coefficient Extraction

For each sampled entry:

1. A future horizon of candles is analyzed.
2. The minimum future low is extracted.
3. A Stop Loss coefficient is calculated:

```python
coef_sl = (entry_price - future_min_low) / ATR
```

This produces a distribution of empirical SL multipliers based on observed price behavior.

---

## 3️⃣ Phase 2 — Buying Power Constrained Optimization

The objective is to find the SL coefficient that satisfies a capital constraint.

Position sizing is computed as:

```python
shares = risk_per_trade / (coef_sl * ATR)
buying_power = entry_price * shares
```

Where:

- Risk per trade = 100 USD
- Desired Buying Power range = 1200–1500 USD

The algorithm:

- Iterates through sorted candidate SL coefficients
- Computes average Buying Power
- Selects the first coefficient within range
- If none match, selects the one closest to the midpoint

This ensures volatility-adjusted risk control while maintaining capital efficiency.

---

## 4️⃣ Phase 3 — Multi-Ratio TP/SL Simulation

Once the SL coefficient is selected:

Take Profit levels are defined as:

```python
TP = ratio * coef_sl * ATR
```

Tested ratios:

- 1.0x
- 1.5x
- 2.0x

For each entry:

- The script scans future candles
- Detects the first hit (TP or SL)
- Measures duration in candles

The result includes:

- Average duration per ratio
- Selected SL coefficient
- Average Buying Power

---

# Key Parameters

These can be modified directly in the script:

```
ATR_PERIOD = 50
HORIZONTE_VELAS = 20
MUESTRAS_POR_ACTIVO = 500
RIESGO_POR_TRADE = 100
BP_MIN = 1200
BP_MAX = 1500
RATIOS_TP_SL = [1.0, 1.5, 2.0]
```

---

# What This Project Demonstrates

- Volatility normalization using ATR
- Empirical risk calibration
- Position sizing based on fixed dollar risk
- Capital constraint optimization
- Vectorized event detection for TP/SL hits

---

# Requirements

```
pandas
numpy
```

---

# Disclaimer

This project is for research and educational purposes only.  
It does not constitute financial advice.
