import gurobipy
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt

market_df = pd.read_csv('market-return.csv')
market_df = market_df.dropna()
monthly_stock_returns = pd.read_csv('monthly-return-capitalization.csv')
monthly_stock_returns = monthly_stock_returns.dropna()
dates = list(set(monthly_stock_returns['Monthly Calendar Date']))
dates.sort()
#print(dates)
date_to_num = {}
for i in range(len(dates)):
    date_to_num[dates[i]] = i
date_ub = len(dates)
monthly_stock_returns['Monthly Calendar Date'] = monthly_stock_returns['Monthly Calendar Date'].apply(lambda k: date_to_num[k])
#print(monthly_stock_returns.head)
# print(monthly_stock_returns.columns)
# print((set(monthly_stock_returns['Ticker'])))
# stock_universe = list(set(monthly_stock_returns['Ticker']))
# stock_universe.remove('V')
stock_universe =['PG', 'NFLX', 'CSCO', 'UNH', 'MCD', 'JPM', 'AMZN', 'PM', 'INTC', 'CRM', 'TXN', 'AMD', 'PEP', 'PFE',
                 'ADBE', 'TMO', 'NVDA', 'MSFT', 'AVGO', 'DIS', 'XOM', 'HD', 'BAC', 'CAT', 'JNJ', 'AMGN',
                 'CMCSA', 'INTU', 'LLY', 'WMT']

stock_returns = {}
for stock in stock_universe:
    stock_df = monthly_stock_returns[(monthly_stock_returns['Ticker'] == stock)]
    stock_returns[stock] = list(stock_df['Monthly Total Return'])
#print(stock_returns['PG'][1])
# print(stock_universe[-1])
# print(stock_universe)
# stock_dataframes_ticker = {}
# full_length_univ = []
# for stock in stock_universe:
#     stock_dataframes_ticker[stock] = monthly_stock_returns.loc[monthly_stock_returns['Ticker'] == stock]
#     stock_dataframes_ticker[stock] = stock_dataframes_ticker[stock].drop(['Header CUSIP -8 Characters', 'PERMCO',
#                                                                           'Monthly Market Capitalization',
#                                                                           'Shares Outstanding'], axis=1)
#     stock_dataframes_ticker[stock]['Monthly Calendar Date'] = range(len(list(stock_dataframes_ticker[stock]['Monthly Calendar Date'])))
# print((full_length_univ[:30]))
# for stock in stock_universe:
#     print(stock_dataframes_ticker[stock].head)

def constant_corr_cov(df, stock_univ, current_date, past_extension):
    df_filtered = df[(df['Monthly Calendar Date'] >= current_date - past_extension) &
                     (df['Monthly Calendar Date'] <= current_date) &
                     (df['Ticker'].isin(stock_univ))]
    pivot_table = df_filtered.pivot_table(index='Monthly Calendar Date', columns='Ticker',
                                          values='Monthly Total Return')
    variance = pivot_table.var(ddof=1)
    covariance_matrix = pivot_table.cov()
    correlation_matrix = pivot_table.corr()
    constant_correlation = (correlation_matrix.sum().sum() - 30) / (30 ** 2 - 30)
    std_dev = np.sqrt(variance)
    sigma_matrix = np.outer(std_dev, std_dev)
    diag_matrix = np.diag(variance)
    v = constant_correlation * sigma_matrix + (1 - constant_correlation) * diag_matrix
    return v

def sample_cov_with_shrinkage_simple(df, stock_univ, current_date, past_extension):
    ccc = constant_corr_cov(df, stock_univ, current_date, past_extension)
    df_filtered = df[(df['Monthly Calendar Date'] >= current_date - past_extension) &
                     (df['Monthly Calendar Date'] <= current_date) &
                     (df['Ticker'].isin(stock_univ))]
    pivot_table = df_filtered.pivot_table(index='Monthly Calendar Date', columns='Ticker',
                                          values='Monthly Total Return')
    sample_cov = pivot_table.cov().to_numpy()
    return .8*sample_cov + .2*ccc

def sample_cov_single_factor(df, stock_univ, current_date, past_extension, mdf):
    df_filtered = df[(df['Monthly Calendar Date'] >= current_date - past_extension) &
                     (df['Monthly Calendar Date'] <= current_date) &
                     (df['Ticker'].isin(stock_univ))]

    # Pivot the table to have dates as rows and stocks as columns
    pivot_table = df_filtered.pivot_table(index='Monthly Calendar Date', columns='Ticker',
                                          values='Monthly Total Return')
    # Align market returns with the pivot table
    market_returns = mdf.set_index('Date')['Total Market']

    # Initialize arrays for factor loadings, residual returns, and expected returns
    beta_i = np.zeros(len(stock_univ))
    theta_i = np.zeros(pivot_table.shape)
    expected_returns_i = np.zeros(len(stock_univ))

    # Calculate beta_i, theta_i, and expected_returns_i for each stock
    for i, stock in enumerate(stock_univ):
        r_i = pivot_table[stock].values
        r_M = market_returns.values
        # Ensure the length matches
        min_len = min(len(r_i), len(r_M))
        r_i = r_i[:min_len]
        r_M = r_M[:min_len]
        r_M = np.array([float(r[:-1]) for r in r_M])
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
    return cov_matrix




# start_date = date_to_num['2017-01-31']
# past_ext = 84
# V = constant_corr_cov(monthly_stock_returns, stock_universe, date_to_num[start_date], past_ext)
# print(V)
def min_variance_portfolio(cov_matrix, max_constr, univ):
    univ_length = len(univ)
    min_var_model = gurobipy.Model()
    min_var_model.Params.LogToConsole = 0
    x = min_var_model.addMVar(univ_length, ub=max_constr)
    # print(np.array([1,1]) @ x)
    # print(np.array(cov_matrix @ x))
    min_var_model.update()
    s1 = cov_matrix @ x
    obj = s1 @ x
    min_var_model.setObjective(obj , gurobipy.GRB.MINIMIZE)
    c1 = min_var_model.addConstr(sum(x) == 1, "Norm")
    min_var_model.optimize()
    return np.array(min_var_model.X), min_var_model.objVal

# test_cov = np.array([np.array([1,0]),
#                     np.array([0,10])])
# print(min_variance_portfolio([], test_cov, .8, ["A", "B"]))

# init_min_port, min_port_var = min_variance_portfolio(V, .08, stock_universe)
# print(init_min_port)
def portfolio_return(portfolio, date, stock_ret, univ):
    ret = 0
    for i in range(len(univ)):
        stock = univ[i]
        ret += portfolio[i] * stock_ret[stock][date]
    return ret


def measure_long_term_performance_const_corr(raw_df, start_date, past_ext, conc_const, univ, stock_ret, last_date):
    if conc_const < 1/30:
        return "Constraint Error"
    curr_date = start_date
    portfolio_performance = []
    equal_benchmark_performance = []
    equal_benchmark_portfolio = np.array([1/len(univ) for i in range(len(univ))])
    portfolios = []
    curr_portfolio = []
    supposed_variances = []
    curr_var = 0
    while curr_date < last_date:
        if (curr_date - start_date) % 6 == 0:
            cov_estimate = constant_corr_cov(raw_df, univ, curr_date, past_ext)
            curr_portfolio, curr_var = min_variance_portfolio(cov_estimate, conc_const, univ)
            portfolios.append(curr_portfolio)
            supposed_variances.append(curr_var)
        if curr_date > start_date:
            portfolio_performance.append(portfolio_return(curr_portfolio, curr_date, stock_ret, univ))
            equal_benchmark_performance.append(portfolio_return(equal_benchmark_portfolio, curr_date, stock_ret, univ))
        curr_date += 1

    return portfolio_performance, equal_benchmark_performance, portfolios, supposed_variances


def measure_long_term_performance_sampl_shrink(raw_df, start_date, past_ext, conc_const, univ, stock_ret, last_date):
    if conc_const < 1/30:
        return "Constraint Error"
    curr_date = start_date
    portfolio_performance = []
    equal_benchmark_performance = []
    equal_benchmark_portfolio = np.array([1/len(univ) for i in range(len(univ))])
    portfolios = []
    curr_portfolio = []
    supposed_variances = []
    curr_var = 0
    while curr_date < last_date:
        if (curr_date - start_date) % 6 == 0:
            cov_estimate = sample_cov_with_shrinkage_simple(raw_df, univ, curr_date, past_ext)
            curr_portfolio, curr_var = min_variance_portfolio(cov_estimate, conc_const, univ)
            portfolios.append(curr_portfolio)
            supposed_variances.append(curr_var)
        if curr_date > start_date:
            portfolio_performance.append(portfolio_return(curr_portfolio, curr_date, stock_ret, univ))
            equal_benchmark_performance.append(portfolio_return(equal_benchmark_portfolio, curr_date, stock_ret, univ))
        curr_date += 1

    return portfolio_performance


def measure_long_term_performance_single_factor(raw_df, start_date, past_ext, conc_const, univ, stock_ret, last_date, mdf):
    if conc_const < 1/30:
        return "Constraint Error"
    curr_date = start_date
    portfolio_performance = []
    equal_benchmark_performance = []
    equal_benchmark_portfolio = np.array([1/len(univ) for i in range(len(univ))])
    portfolios = []
    curr_portfolio = []
    supposed_variances = []
    curr_var = 0
    while curr_date < last_date:
        if (curr_date - start_date) % 6 == 0:
            cov_estimate = sample_cov_single_factor(raw_df, univ, curr_date, past_ext, mdf)
            curr_portfolio, curr_var = min_variance_portfolio(cov_estimate, conc_const, univ)
            portfolios.append(curr_portfolio)
            supposed_variances.append(curr_var)
        if curr_date > start_date:
            portfolio_performance.append(portfolio_return(curr_portfolio, curr_date, stock_ret, univ))
            equal_benchmark_performance.append(portfolio_return(equal_benchmark_portfolio, curr_date, stock_ret, univ))
        curr_date += 1

    return portfolio_performance

perf, ben, port, svar = measure_long_term_performance_const_corr(monthly_stock_returns, 84, 60, .08, stock_universe,
                                                           stock_returns, date_ub)
ssh = measure_long_term_performance_sampl_shrink(monthly_stock_returns, 84, 84, .08, stock_universe,
                                                           stock_returns, date_ub)
sf = measure_long_term_performance_single_factor(monthly_stock_returns, 84, 84, .08, stock_universe,
                                                           stock_returns, date_ub, market_df)

plt.plot(perf)
plt.plot(ssh)
plt.plot(sf)
plt.plot(ben)
plt.legend(['Min Var (const. corr.)','Min Var (Sample Shrinkage, intensity .5)','Single-Factor', 'Equal'])
plt.xlabel('Months since January 2017')
plt.ylabel('Return (%)')
print(np.var(perf), np.var(ssh), np.var(sf), np.var(ben))
print(np.mean(perf), np.mean(ssh), np.mean(sf), np.mean(ben))
plt.savefig("realquickgraph.png")
plt.show()





