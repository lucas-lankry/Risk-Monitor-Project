![Python](https://img.shields.io/badge/Python-3.9%2B-blue)
![Streamlit](https://img.shields.io/badge/Streamlit-App-red)
![Bloomberg](https://img.shields.io/badge/Bloomberg-API-black)
[![Demo Video](https://img.shields.io/badge/Demo-Video-blueviolet)](Tutorial%20Risk%20Monitor%20App.mp4)


# Risk Monitor Project

## Overview

This project is a **real-time risk monitoring application for commodity options trading**, built with **Python** and **Streamlit**, and live integration to **Bloomberg**.

The core idea is simple:

> **Standardize the way executed trades are stored (via an Excel file), and transform that static trade blotter into a dynamic, real-time risk monitor using Bloomberg market data and multiple pricing models.**

The application gives:

* A clear, consolidated view of Greeks and PnL
* Model comparison beyond Bloomberg’s built-in analytics
* Fast visualization and export of risk metrics

📹 **Application Demo Video**

The repository includes a short video demonstrating the application in action:

![Risk Monitor Demo](Tutorial%20Risk%20Monitor%20App.mp4)


* **File**: `Tutorial Risk Monitor App.mp4`
* The video walks through:

  * Loading the Excel trade blotter
  * Visualizing real-time Greeks and risk
  * Exploring volatility smiles and model outputs
  * Exporting and emailing risk reports

---

## Trade Input: Standardized Excel Blotter

All trades are assumed to be recorded in a **standardized Excel file** ("Commodities_Deals.xlsx")containing:

* Option tickers
* Contract Size
* Position sizes (long / short)
* Strikes
* Maturities
* Trade prices
* Option type (Call / Put)

The application allows you to:

* Upload this Excel file directly
* Parse and clean the data
* Use it as the basis for all downstream risk calculations

This approach ensures consistency, reproducibility, and easy integration with existing trading workflows.

---

## Bloomberg Integration (Requirement)

⚠️ **A running Bloomberg Terminal is required**

This project relies on the **Bloomberg API (`blpapi`)**, which means:

* You must have a **Bloomberg Terminal installed**
* The terminal **must be open and running** while the app is used
* The Bloomberg API service must be accessible locally (default port: `8194`)

Bloomberg is used to fetch:

* Option prices (Bid / Ask / Last)
* Implied volatilities
* Greeks (Delta, Gamma, Vega, Theta, Rho)
* Underlying futures prices
* US Treasury yields (govies) for rate curve construction

The Python API can be installed through ‘pip’ using:

>> python -m pip install --index-url=https://blpapi.bloomberg.com/repository/releases/python/simple/ blpapi

---

## Real-Time Risk Monitoring

Once the trade file is loaded and Bloomberg data is available, the application builds a **real-time risk monitor**, including:

* Live market prices
* Updated Greeks per position
* Position-level PnL
* Aggregated portfolio risk

The goal is to replicate — and extend — a professional trading desk risk view in a lightweight, customizable application.

---

## Pricing Models & Greeks

A key differentiator of this project is the ability to **compare Greeks across multiple models**, not just Bloomberg.

### Implemented Models

#### 1. Bloomberg Greeks

* Directly retrieved via `blpapi`
* Used as a reference / benchmark

#### 2. Black-76 (B76)

* Standard model for options on futures
* Uses forward prices and discounting
* Greeks computed analytically

#### 3. Bachelier (Normal Model)

* Useful for low-price or near-zero underlyings
* Greeks computed analytically

#### 4. Heston Stochastic Volatility Model

* Stochastic variance framework
* Prices computed via characteristic functions
* Greeks computed via finite differences

#### 5. Local Volatility Model

* Built from market implied volatility smiles
* Uses Dupire-style intuition
* Allows strike- and maturity-dependent volatility

---

## Interest Rate Curve (US Govies)

Instead of using a flat risk-free rate, the application builds a **term structure of interest rates** using:

* US Treasury yields (1M → 30Y)
* Bloomberg government bond tickers

For each option maturity:

* The corresponding rate is interpolated from the curve
* The correct maturity-specific discount rate is applied

This improves pricing accuracy, especially for longer-dated options.

---

## Risk Visualization

### Bloomberg, Black-76 & Bachelier

For these models, the app provides **two complementary visualizations**:

1. **Greeks by Position**

   * Delta, Gamma, Vega, Theta, Rho per trade
   * Clear identification of dominant risk contributors

2. **Aggregated Greeks by Maturity**

   * Greeks summed across positions
   * Bucketed by maturity
   * Helps identify concentration of risk over time

---

### Heston & Local Volatility

For more advanced models, a different approach is used.

Instead of focusing directly on Greeks, the app emphasizes **volatility structure**:

#### Volatility Smile Visualization

* Interactive volatility smile graphs
* Selection menu to choose the **underlying commodity**
* Display of model-implied vs market-implied volatility

#### Market Data Enrichment

For a given underlying and maturity:

* The app can automatically query Bloomberg
* Fetches **options with the same maturity but different strikes**
* Builds a richer volatility smile from market data

This allows:

* Model validation
* Detection of skew and smile effects
* Comparison between theoretical and market-implied volatility

---

## Export & Reporting

The project includes a full **export and reporting pipeline**.

You can:

* Export the complete risk analysis to **Excel**
* Export to **CSV** for downstream systems
* Send the Excel report **by email** directly from the app

 **Email note**:

* Emails are sent via **SendGrid**
* Attachments are included
* Messages may frequently end up in **spam folders** depending on email provider settings

---

## Technology Stack

* **Python**
* **Streamlit** – interactive web interface
* **Bloomberg API (`blpapi`)** – market data
* **Pandas / NumPy** – data handling
* **SciPy** – numerical methods
* **Plotly** – interactive charts
* **SendGrid** – email delivery

---

## Disclaimer

This application is for **risk analysis and visualization purposes only**.
It is **not intended for automated trading** or execution.

All pricing models are provided "as is" and should be validated before being used for any real trading decisions.
