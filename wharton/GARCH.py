import os
import numpy as np 
import pandas as pd 
import seaborn as sns 
from arch import arch_model 
from scipy.stats import zscore 
import matplotlib.pyplot as plt 
from statsmodels.tsa.stattools import adfuller 
from statsmodels.tsa.seasonal import seasonal_decompose 
from sklearn.metrics import mean_squared_error, mean_absolute_error
from statsmodels.tsa.seasonal import seasonal_decompose
from scipy.stats import zscore

# Force Matplotlib to use an interactive desktop backend (Qt or Tk)
try:
    import matplotlib
    matplotlib.use("TkAgg")
except Exception:
    pass

# --- 1. Load Data ---
script_dir = os.path.dirname(os.path.abspath(__file__))
file_path = os.path.join(script_dir, "market_data.csv")

data = pd.read_csv(file_path)

# --- 2. Fix Date Parsing & Types ---
data["Date"] = pd.to_datetime(data["Date"], format="mixed", errors="coerce")
data["Close"] = pd.to_numeric(data["Close"], errors="coerce")
data["High"] = pd.to_numeric(data["High"], errors="coerce")
data["Low"] = pd.to_numeric(data["Low"], errors="coerce")

# Drop invalid rows and set Date index
data = data.dropna(subset=["Date", "Close"])
data = data.set_index("Date").sort_index()

# Calculate returns
data["returns"] = 100 * data["Close"].pct_change()


# --- 3. Initial Plotting Function ---
def plot_data():
    fig, axes = plt.subplots(3, 1, figsize=(10, 8))

    data["Close"].plot(
        ax=axes[0], title="Adjusted Close Price", color="blue", ylabel="Price"
    )

    axes[1].plot(data.index, data["High"], label="High", color="blue")
    axes[1].plot(data.index, data["Low"], label="Low", color="orange")
    axes[1].set_title("High and Low Prices")
    axes[1].set_ylabel("Price")
    axes[1].legend()

    data["returns"].dropna().plot(
        ax=axes[2],
        title="Daily Returns (%)",
        color="green",
        ylabel="Returns (%)",
    )

    plt.tight_layout()
    plt.show(block=True)


plot_data()


# --- 4. ADF Test ---
def adf_test(series):
    result = adfuller(series.dropna())
    print(f"ADF Statistic: {result[0]:.4f}")
    print(f"p-value: {result[1]:.4f}")
    return result[1]


adf_pvalue = adf_test(data["returns"])

if adf_pvalue > 0.05:
    data["stationarity_returns"] = data["returns"].diff()
else:
    data["stationarity_returns"] = data["returns"]


# --- 5. Plot Rolling Volatility ---
rolling_vol = data["stationarity_returns"].rolling(window=30).std().dropna()

print(f"Number of valid rolling volatility rows: {len(rolling_vol)}")

if len(rolling_vol) > 0:
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(rolling_vol.index, rolling_vol.values, color="red")
    ax.set_title("Rolling Volatility (30 days)")
    ax.set_xlabel("Date")
    ax.set_ylabel("Volatility")
    plt.tight_layout()
    plt.show(block=True)
else:
    print("No data available to plot rolling volatility.")

# Correlation matrix heatmap
corr_matrix = data[['Close', 'Prev Close', 'High', 'Low', 'returns']].corr()
plt.figure(figsize=(8, 6))
sns.heatmap(corr_matrix, annot=True, cmap= 'coolwarm', fmt='.2f')
plt.title( 'Correlation Heatmap')
plt.show()

# Distribution of daily returns
plt.figure(figsize=(10, 6))
sns.histplot(data['returns'], bins=50, kde=True)
plt.title('Distribution of Daily Returns')
plt.xlabel ('Returns')
plt.ylabel ('Frequency')
plt.show()


returns_no_na = data["returns"].dropna()
data["z_score"] = pd.Series(zscore(returns_no_na), index=returns_no_na.index)
outliers = data[data["z_score"].abs() > 3]
print(f"Number of outliers detected: {len(outliers)}")

# Plot outliers on top of returns
plt. figure(figsize=(10, 6))
plt.plot(data[ 'returns'], label= 'Daily Returns')
plt.scatter(outliers.index, outliers['returns'], color='red', label='Outliers')
plt. title( 'Daily_Returns_with_Outliers')
plt. xlabel( 'Date')
plt.ylabel('Returns')
plt.legend() 
plt.show()

# Detect and cap outliers
def cap_outliers(series):
    lower_bound = series.quantile(0.01)
    upper_bound = series. quantile(0.99)
    return series.clip(lower=lower_bound, upper=upper_bound)
data[ 'returns'] = cap_outliers(data['returns'])

# Plot the daily returns after capping the outliers
plt.figure(figsize=(10, 6))
plt.plot(data[ 'returns'],label='Capped Daily Returns') 
plt.title('Daily_Returns_with_Outliers_Capped')
plt.xlabel('Date')
plt.ylabel('Returns')
plt.legend()
plt.show()

# Decompose the 'Close' prices
n = len(data["Close"].dropna())

# Candidate cycles, largest first: yearly, quarterly, monthly, weekly (trading days)
candidates = {"yearly": 252, "quarterly": 63, "monthly": 21, "weekly": 5}

for name, period in candidates.items():
    if n >= 2 * period:
        break
else:
    raise ValueError(f"Only {n} rows, not enough to decompose")

print(f"Using {name} cycle (period={period}) with {n} observations")

decomposed = seasonal_decompose(data["Close"].dropna(), model="multiplicative", period=period)
decomposed.plot()
plt.suptitle(f"Seasonal decomposition ({name}, period={period})")
plt.show()

# Rolling Mean and Standard Deviation
rolling_mean = data[ 'Close'].rolling(window=30).mean()
rolling_std = data[ 'Close'].rolling(window=30).std()
plt.figure(figsize=(10, 6))
plt.plot(data[ 'Close'], label='Close Price', color='blue') 
plt.plot(rolling_mean, label='Rolling Mean (30 days)', color='red') 
plt.plot(rolling_std, label='Rolling Std Dev (30 days)', color='green') 
plt.title('Rolling Mean and Standard Deviation - Close Price')
plt.legend()
plt.show()

def fit_models (series):
    arch_model_fit = arch_model(series.dropna(), vol='ARCH', p=1).fit()
    print("ARCH Model Summary")
    print(arch_model_fit.summary())
    garch_model_fit = arch_model(series.dropna(), vol='Garch', p=1, q=1).fit()
    print("GARCH Model Summary")
    print(garch_model_fit.summary())
    return arch_model_fit, garch_model_fit
arch_fit, garch_fit = fit_models(data['returns'])

def forecast_volatility(model, horizon=5):
    forecast = model.forecast(horizon=horizon)
    return np.sqrt(forecast.variance.iloc[-1])
arch_forecast_volatility = forecast_volatility(arch_fit)
garch_forecast_volatility = forecast_volatility(garch_fit)

# Plot forecasted volatility for both models
plt.figure(figsize=(10, 6))
plt.plot(arch_forecast_volatility, label='Forecasted Volatility (ARCH)', color='blue') 
plt.plot(garch_forecast_volatility, label='Forecasted Volatility (GARCH)', color='purple')
plt.title('Forecasted Volatility Comparison')
plt.xlabel('Days')
plt.ylabel('Volatility')
plt.legend()
plt.show()

print(f"ARCH Model AIC: {arch_fit.aic}")
print(f"ARCH Model BIC: {arch_fit.bic}")
print(f"GARCH Model AIC: {garch_fit.aic}")
print(f"GARCH Model BIC: {garch_fit.bic}")

actual_volatility = data[ 'returns'].rolling(window=5).std().dropna()

def calculate_errors(actual, forecasted):
    mse = mean_squared_error (actual, forecasted)
    mae = mean_absolute_error (actual, forecasted)
    return mse, mae

# ARCH model MSE and MAE
arch_mse, arch_mae = calculate_errors(actual_volatility.iloc[-5:], arch_forecast_volatility)
print(f"ARCH Model MSE: {arch_mse}, MAE: {arch_mae}")

#GARCH model MSE and MAE
garch_mse, garch_mae = calculate_errors(actual_volatility.iloc[-5:], garch_forecast_volatility)
print(f"GARCH Model MSE: {garch_mse}, MAE: {garch_mae}")

plt.figure(figsize=(10,6))
plt.plot(["ARCH", "GARCH"], [arch_mse, garch_mse], marker="o", label="MSE")
plt.plot(["ARCH", "GARCH"], [arch_mae, garch_mae], marker="o", label="MAE")
plt.title ('Model_Performance_Comparison')
plt.xlabel ('Model')
plt.ylabel('Error')
plt.legend()
plt.show()