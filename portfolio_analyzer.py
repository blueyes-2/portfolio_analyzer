import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import yfinance as yf

# 1. Collect Data
tickers = ["AAPL", "MSFT", "GOOG", "TSLA", "TLT", "GLD", "^GSPC"]
data = yf.download(tickers, start="2019-01-01", end="2024-01-01")  # 5 years

# Use 'Close' prices
close_prices = data.loc[:, ("Close",)]

# 2. Calculate Returns
returns = close_prices.pct_change().dropna()

# 3. Volatility (Annualized Standard Deviation)
volatility = returns.std() * np.sqrt(252)
print("Volatility:\n", volatility)

# 4. Correlation Matrix (Static)
corr_matrix = returns.corr()
sns.heatmap(corr_matrix, annot=True, cmap="coolwarm")
plt.title("Correlation Heatmap")
plt.show()

# 5. Portfolio Simulation
weights = np.array([0.15, 0.15, 0.15, 0.15, 0.15, 0.15, 0.10])
portfolio_return = np.sum(weights * returns.mean()) * 252
cov_matrix = returns.cov() * 252
portfolio_volatility = np.sqrt(np.dot(weights.T, np.dot(cov_matrix, weights)))

print("Portfolio Return:", portfolio_return)
print("Portfolio Volatility:", portfolio_volatility)

# 6. Sharpe Ratio
risk_free_rate = 0.02
sharpe_ratio = (portfolio_return - risk_free_rate) / portfolio_volatility
print("Sharpe Ratio:", sharpe_ratio)

# 7. Visualization - Static Volatility
plt.bar(volatility.index, volatility.values)
plt.title("Annualized Volatility")
plt.ylabel("Volatility")
plt.show()

# 8. Rolling Volatility (Dynamic Risk)
rolling_vol = returns.rolling(60).std() * np.sqrt(252)
rolling_vol.plot(figsize=(10,6))
plt.title("Rolling 60-Day Annualized Volatility")
plt.ylabel("Volatility")
plt.legend(rolling_vol.columns, loc="upper left")
plt.show()

# 9. Rolling Correlation (Dynamic Diversification)
# Example: correlation between AAPL (stocks) and TLT (bonds)
rolling_corr = returns['AAPL'].rolling(60).corr(returns['TLT'])
rolling_corr.plot(figsize=(10,6))
plt.title("Rolling 60-Day Correlation: AAPL vs TLT")
plt.ylabel("Correlation")
plt.axhline(0, color='black', linestyle='--')  # reference line
plt.show()

# Another example: correlation between S&P 500 (^GSPC) and Gold (GLD)
rolling_corr2 = returns['^GSPC'].rolling(60).corr(returns['GLD'])
rolling_corr2.plot(figsize=(10,6), color='gold')
plt.title("Rolling 60-Day Correlation: S&P 500 vs Gold")
plt.ylabel("Correlation")
plt.axhline(0, color='black', linestyle='--')
plt.show()
