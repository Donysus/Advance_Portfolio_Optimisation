# Advanced Portfolio Optimization with Black–Litterman & HRP

This project implements an advanced portfolio optimization framework that integrates the Black–Litterman model with Hierarchical Risk Parity (HRP) to construct a risk-balanced portfolio.

## Overview

The goal of this project is to:
- **Fetch historical market data** for a basket of stocks using yfinance.
- **Compute equilibrium returns** and incorporate user-defined views using the Black–Litterman model.
- **Apply Hierarchical Risk Parity (HRP)** to derive asset weights that balance risk across the portfolio.
- **Visualize** the resulting portfolio weights, expected returns, and portfolio volatility.

This framework is designed to be executed in a Google Colab environment, making it easy to experiment and iterate on quantitative finance strategies.

## Features

- **Data Acquisition:** Downloads adjusted close prices with auto-adjustment using yfinance.
- **Return & Covariance Estimation:** Computes daily returns, annualized returns, and robust covariance matrices (using the Ledoit-Wolf estimator).
- **Black–Litterman Integration:** Adjusts market equilibrium returns with custom views (e.g., AAPL outperforming MSFT).
- **Hierarchical Risk Parity (HRP):** Uses hierarchical clustering to allocate portfolio weights.
- **Visualization:** Plots HRP portfolio weights and calculates portfolio performance metrics.

## Installation

To run this project, you need Python 3 and the following libraries:
- yfinance
- pandas
- numpy
- matplotlib
- scipy
- scikit-learn

You can install the required libraries using:

```bash
pip install yfinance pandas numpy matplotlib scipy scikit-learn

