import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.cluster.hierarchy import linkage
from scipy.spatial.distance import squareform
from sklearn.covariance import LedoitWolf
import warnings

def black_litterman(prior_returns, cov_matrix, P, Q, tau=0.025, omega=None):
    if omega is None:
        omega = np.diag(np.diag(tau * (P @ cov_matrix @ P.T)))
    inv_term = np.linalg.inv(np.linalg.inv(tau * cov_matrix) + P.T @ np.linalg.inv(omega) @ P)
    posterior_returns = inv_term @ (np.linalg.inv(tau * cov_matrix) @ prior_returns + P.T @ np.linalg.inv(omega) @ Q)
    return posterior_returns

def correlDist(corr):
    return np.sqrt(0.5 * (1 - corr))

def get_quasi_diag(link, num_assets):
    sort_ix = [int(link[-1, 0]), int(link[-1, 1])]
    sort_ix = recursive_sort(link, sort_ix, num_assets)
    return sort_ix

def recursive_sort(link, sort_ix, num_assets):
    new_order = []
    for ix in sort_ix:
        if ix < num_assets:
            new_order.append(ix)
        else:
            left = int(link[ix - num_assets, 0])
            right = int(link[ix - num_assets, 1])
            new_order.extend(recursive_sort(link, [left, right], num_assets))
    return new_order

def hrp_allocation(cov, sorted_indices):
    weights = pd.Series(1, index=sorted_indices)
    clusters = [sorted_indices]
    while len(clusters) > 0:
        clusters_new = []
        for cluster in clusters:
            if len(cluster) == 1:
                continue
            split = int(len(cluster) / 2)
            cluster1 = cluster[:split]
            cluster2 = cluster[split:]
            var1 = get_cluster_var(cov, cluster1)
            var2 = get_cluster_var(cov, cluster2)
            alloc_factor = 1 - var1 / (var1 + var2)
            for i in cluster1:
                weights[i] = weights[i] * alloc_factor
            for i in cluster2:
                weights[i] = weights[i] * (1 - alloc_factor)
            clusters_new += [cluster1, cluster2]
        clusters = clusters_new
    return weights.sort_index()

def get_cluster_var(cov, cluster_items):
    sub_cov = cov[np.ix_(cluster_items, cluster_items)]
    inv_diag = 1 / np.diag(sub_cov)
    weights = inv_diag / np.sum(inv_diag)
    cluster_var = np.dot(weights, np.dot(sub_cov, weights))
    return cluster_var

def fetch_price_data(tickers, start_date, end_date):
    data = yf.download(tickers, start=start_date, end=end_date, auto_adjust=True)["Close"]
    data = data.dropna(axis=1, how='all')
    missing_tickers = set(tickers) - set(data.columns)
    if missing_tickers:
        warnings.warn(f"Data for tickers {missing_tickers} not found. They will be dropped.")
    return data

tickers = ["AAPL", "MSFT", "GOOG", "AMZN", "META", "JPM", "BAC", "WMT", "TSLA", "NFLX"]
start_date = "2018-01-01"
end_date = "2023-01-01"
print("Fetching price data...")
price_data = fetch_price_data(tickers, start_date, end_date)
print("Tickers used:", list(price_data.columns))
returns = price_data.pct_change(fill_method=None).dropna()
if returns.empty:
    raise ValueError("Returns data is empty after processing. Check data download or ticker symbols.")
annual_return = returns.mean() * 252
cov_matrix = returns.cov() * 252
lw = LedoitWolf()
lw.fit(returns)
cov_matrix_shrink = pd.DataFrame(lw.covariance_, index=returns.columns, columns=returns.columns)
pi = annual_return.values
n = len(price_data.columns)
P = np.zeros((1, n))
tickers_used = list(price_data.columns)
if "AAPL" in tickers_used and "MSFT" in tickers_used:
    P[0, tickers_used.index("AAPL")] = 1
    P[0, tickers_used.index("MSFT")] = -1
else:
    raise ValueError("Required tickers for the view are missing.")
Q = np.array([0.02])
tau = 0.05
posterior_returns = black_litterman(pi, cov_matrix_shrink.values, P, Q, tau=tau)
bl_returns = pd.Series(posterior_returns, index=tickers_used)
print("\nPosterior Returns from Black-Litterman:")
print(bl_returns)
corr = cov_matrix_shrink.corr().values
dist = correlDist(corr)
link = linkage(squareform(dist), method='single')
sorted_indices = get_quasi_diag(link, n)
hrp_weights = hrp_allocation(cov_matrix_shrink.values, sorted_indices)
hrp_weights.index = [tickers_used[i] for i in hrp_weights.index]
hrp_weights = hrp_weights / hrp_weights.sum()
print("\nHRP Allocation Weights:")
print(hrp_weights)
plt.figure(figsize=(10, 6))
hrp_weights.sort_values().plot(kind='bar', color='teal')
plt.title("Hierarchical Risk Parity (HRP) Portfolio Weights")
plt.xlabel("Asset")
plt.ylabel("Weight")
plt.show()
portfolio_return = np.dot(hrp_weights.values, bl_returns.loc[hrp_weights.index].values)
portfolio_vol = np.sqrt(np.dot(hrp_weights.values, np.dot(cov_matrix_shrink.loc[hrp_weights.index, hrp_weights.index].values, hrp_weights.values)))
print(f"\nOptimized Portfolio Expected Annual Return: {portfolio_return:.2%}")
print(f"Optimized Portfolio Annual Volatility: {portfolio_vol:.2%}")
