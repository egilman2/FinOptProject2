import pandas as pd
import numpy as np
import os

dirname = os.path.dirname(__file__)
# Load the Excel file
file_path = dirname + '/monthly-return-capitalization.xlsx'
df = pd.read_excel(file_path)
path2 = dirname + "/market-return.xlsx"
market_df = pd.read_excel(path2)

# List of specific stocks to consider
selected_stocks = ['PG', 'NFLX', 'CSCO', 'UNH', 'MCD', 'JPM', 'AMZN', 'PM', 'INTC', 'CRM', 'TXN', 'AMD', 'PEP', 'PFE',
                 'ADBE', 'TMO', 'NVDA', 'MSFT', 'AVGO', 'DIS', 'XOM', 'HD', 'BAC', 'CAT', 'JNJ', 'AMGN',
                 'CMCSA', 'INTU', 'LLY', 'WMT']

# Filter the data for dates between 2010 and 2017 and for the selected stocks
df_filtered = df[(df['Monthly Calendar Date'] >= '2010-01-01') & 
                 (df['Monthly Calendar Date'] <= '2017-12-31') &
                 (df['Ticker'].isin(selected_stocks))]

# Pivot the table to have dates as rows and stocks as columns
pivot_table = df_filtered.pivot_table(index='Monthly Calendar Date', columns='Ticker', values='Monthly Total Return')
# Align market returns with the pivot table
market_returns = market_df.set_index('Date')['Total Market']

# Initialize arrays for factor loadings, residual returns, and expected returns
beta_i = np.zeros(len(selected_stocks))
theta_i = np.zeros(pivot_table.shape)
expected_returns_i = np.zeros(len(selected_stocks))

# Calculate beta_i, theta_i, and expected_returns_i for each stock
for i, stock in enumerate(selected_stocks):
    r_i = pivot_table[stock].values
    r_M = market_returns.values

    # Ensure the length matches
    min_len = min(len(r_i), len(r_M))
    r_i = r_i[:min_len]
    r_M = r_M[:min_len]

    # Step 1: Factor Loadings
    beta_i[i] = np.linalg.lstsq(r_M.reshape(-1, 1), r_i, rcond=None)[0]

    # Step 2: Residual Returns
    theta_i[:, i] = r_i - beta_i[i] * r_M

    # Step 3: Expected Return of Each Asset
    expected_returns_i[i] = beta_i[i] * np.mean(r_M) + np.mean(theta_i[:, i])

# Step 4: Covariance Matrix
sigma_M_squared = np.var(r_M)
D = np.diag(np.var(theta_i, axis=0))
cov_matrix = sigma_M_squared * np.outer(beta_i, beta_i) + D

# Print or output the covariance matrix
print(cov_matrix)