import pandas as pd
import numpy as np
# import os
#
# dirname = os.path.dirname(__file__)
# # Load the Excel file
# file_path = dirname + '/monthly-return-capitalization.xlsx'
df = pd.read_csv()

# List of specific stocks to consider
selected_stocks = ['PG', 'NFLX', 'CSCO', 'UNH', 'MCD', 'JPM', 'AMZN', 'PM', 'INTC', 'CRM', 'TXN', 'AMD', 'PEP', 'PFE',
                 'ADBE', 'TMO', 'NVDA', 'MSFT', 'AVGO', 'DIS', 'XOM', 'HD', 'BAC', 'CAT', 'JNJ', 'AMGN',
                 'CMCSA', 'INTU', 'LLY', 'WMT']

# Filter the data for dates between 2010 and 2017 and for the selected stocks
df_filtered = df[(df['Monthly Calendar Date'] >= '2010-01-01') & 
                 (df['Monthly Calendar Date'] <= '2017-12-31') &
                 (df['Ticker'].isin(selected_stocks))]

# Pivot the table to have dates as rows and stocks as columns
pivot_table = df_filtered.pivot_table(index='Monthly Calendar Date', columns='Ticker', values='Monthly Market Capitalization')

# Calculate sample variance for each stock
variance = pivot_table.var(ddof=1)

# Calculate covariance matrix
covariance_matrix = pivot_table.cov()

# Calculate correlation matrix
correlation_matrix = pivot_table.corr()

# Display the variance for each stock
variance_display = variance.sort_values(ascending=False).head(10)  # Display top 10 for brevity

# Display a portion of the correlation matrix
correlation_matrix_display = correlation_matrix.iloc[:10, :10]  # Display a 10x10 portion for clarity

print(correlation_matrix)
constant_correlation = (correlation_matrix.sum().sum() - 30) / (30**2 - 30)

# Displaying the results
print("Sample Variance")
print(variance)
print("\nestimated constant correlation")
print(constant_correlation)

# Calculate standard deviation (σ)
std_dev = np.sqrt(variance)

# Create the matrix σσ^T
sigma_matrix = np.outer(std_dev, std_dev)
print(sigma_matrix)

# Create the diagonal matrix with the variances
diag_matrix = np.diag(variance)

# Calculate V
V = constant_correlation * sigma_matrix + (1 - constant_correlation) * diag_matrix
