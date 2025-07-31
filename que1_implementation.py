import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from sklearn.metrics import r2_score
import os
import glob

def power_law_impact_model(x, alpha, beta):
    """Core power-law function: g(x) = alpha * x^beta"""
    return alpha * np.power(x + 1e-9, beta)

def get_impact_params_from_snapshot(snapshot):
    """
    Takes a single row of the order book, calculates empirical slippage,
    and returns the fitted alpha and beta parameters.
    """
    mid_price = (snapshot['ask_px_00'] + snapshot['bid_px_00']) / 2.0
    fixed_spread_cost = snapshot['ask_px_00'] - mid_price

    ask_prices = [snapshot[f'ask_px_{i:02d}'] for i in range(10)]
    ask_sizes = [snapshot[f'ask_sz_{i:02d}'] for i in range(10)]

    order_sizes, empirical_slippage = [], []
    cumulative_shares, cumulative_cost = 0, 0

    for i in range(10):
        if pd.isna(ask_prices[i]) or ask_sizes[i] == 0:
            continue
        
        cumulative_cost += ask_prices[i] * ask_sizes[i]
        cumulative_shares += ask_sizes[i]
        
        slippage = (cumulative_cost / cumulative_shares) - mid_price
        order_sizes.append(cumulative_shares)
        empirical_slippage.append(slippage)

    if len(order_sizes) < 2:
        return np.nan, np.nan, np.nan

    model_fitting_slippage = np.array(empirical_slippage) - fixed_spread_cost

    try:
        params, _ = curve_fit(
            power_law_impact_model,
            order_sizes,
            model_fitting_slippage,
            p0=[0.0001, 0.5],
            maxfev=5000
        )
        fitted_vals = power_law_impact_model(np.array(order_sizes), *params)
        r_squared = r2_score(model_fitting_slippage, fitted_vals)
        return params[0], params[1], r_squared
    except RuntimeError:
        return np.nan, np.nan, np.nan

def process_daily_file(file_path):
    """
    Loads a full day's CSV file and returns a DataFrame of the
    fitted parameters over time at 1-minute intervals.
    """
    print(f"Processing file: {os.path.basename(file_path)}...")
    df = pd.read_csv(file_path)
    df['ts_event'] = pd.to_datetime(df['ts_event'], format='ISO8601')
    df = df.set_index('ts_event').tz_convert('US/Eastern')
    
    trading_start = df.index.min().normalize() + pd.Timedelta(hours=9, minutes=30)
    trading_end = df.index.min().normalize() + pd.Timedelta(hours=16)
    
    analysis_times = pd.date_range(start=trading_start, end=trading_end, freq='1min')
    
    results = []
    for timestamp in analysis_times:
        snapshot = df.asof(timestamp)
        if pd.isna(snapshot.any()):
            continue
            
        alpha, beta, r_squared = get_impact_params_from_snapshot(snapshot)
        results.append({'timestamp': timestamp, 'alpha': alpha, 'beta': beta, 'r_squared': r_squared})
        
    return pd.DataFrame(results)

if __name__ == "__main__":
    ticker_folders = ['FROG', 'CRWV', 'SOUN']
    all_ticker_data = {}

    for ticker in ticker_folders:
        if not os.path.isdir(ticker):
            print(f"Warning: Directory '{ticker}/' not found. Skipping.")
            continue
            
        csv_files = glob.glob(os.path.join(ticker, '*.csv'))
        if not csv_files:
            print(f"No CSV files found in '{ticker}/'. Skipping.")
            continue

        daily_dfs = [process_daily_file(f) for f in csv_files]
        if daily_dfs:
            combined_df = pd.concat(daily_dfs)
            all_ticker_data[ticker] = combined_df
            output_path = f"{ticker}_impact_parameters.csv"
            combined_df.to_csv(output_path, index=False)
            print(f"Saved results to {output_path}")

    print("\nPlotting combined results...")
    plt.style.use('seaborn-v0_8-whitegrid')
    fig, ax = plt.subplots(figsize=(15, 8))

    for ticker, df in all_ticker_data.items():
        if not df.empty:
            # For plotting, we only care about the time of day, not the specific date
            df['time_of_day'] = df['timestamp'].dt.time
            # Calculate the average alpha for each minute of the day across all days
            average_alpha_by_time = df.groupby('time_of_day')['alpha'].mean()
            
            ax.plot(average_alpha_by_time.index.astype(str), average_alpha_by_time.values, label=f'{ticker} (Avg. Alpha)')

    ax.set_title('Average Market Liquidity Profile Throughout the Trading Day', fontsize=16)
    ax.set_xlabel('Time of Day (US/Eastern)', fontsize=12)
    ax.set_ylabel('Average Alpha (α) - Lower is More Liquid', fontsize=12)
    
    tick_positions = np.linspace(0, len(ax.get_xticklabels()) - 1, 10, dtype=int)
    ax.set_xticks(tick_positions)
    ax.tick_params(axis='x', rotation=45)
    
    ax.legend()
    ax.set_yscale('log')
    plt.tight_layout()
    plt.show()