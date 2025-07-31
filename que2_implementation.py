import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

def get_cost_function(alpha, beta, gamma):
    """
    Creates a vectorized cost function for a given set of impact parameters.
    This function calculates C(x) = gamma*x + alpha*x^(1+beta)
    """
    beta_plus_1 = 1 + beta
    def cost_function(x):
        return (gamma * x) + (alpha * np.power(x, beta_plus_1))
    return cost_function

def run_execution_optimizer(impact_params_df, total_shares_to_buy, lot_size=100):
    """
    Finds the optimal execution schedule using Dynamic Programming.
    """
    print(f"\nRunning Optimizer for {total_shares_to_buy} shares")
    print(f"Discretizing share space into lots of {lot_size}")

    S = total_shares_to_buy // lot_size
    
    N = len(impact_params_df)
    
    # Initialize DP tables, V stores the minimum cost, P stores the optimal number of lots to trade at each state
    V = np.full((N + 1, S + 1), np.inf)
    P = np.full((N + 1, S + 1), 0, dtype=int)

    # Base case: cost to buy 0 shares in 0 periods is 0
    V[0, 0] = 0

    for i in range(1, N + 1):
        params = impact_params_df.iloc[i-1]
        cost_func = get_cost_function(params['alpha'], params['beta'], params['gamma'])
        for j in range(S + 1):
            for k in range(j + 1):
                current_cost = cost_func(k * lot_size) + V[i-1, j-k] 
                if current_cost < V[i, j]:
                    V[i, j] = current_cost
                    P[i, j] = k

    if V[N, S] == np.inf:
        print("Error: No solution found. The cost is infinite.")
        return None, np.inf

    optimal_schedule = np.full(N, 0, dtype=int)
    shares_remaining = S
    
    for i in range(N, 0, -1):
        lots_to_trade = P[i, shares_remaining]
        optimal_schedule[i-1] = lots_to_trade * lot_size
        shares_remaining -= lots_to_trade

    total_cost = V[N, S]
    return optimal_schedule, total_cost

def plot_schedule(schedule_df):
    print("Aggregating schedule by time of day for visualization...")
    schedule_df['timestamp'] = pd.to_datetime(schedule_df['timestamp'])
    schedule_df['time_of_day'] = schedule_df['timestamp'].dt.time
    aggregated_schedule = schedule_df.groupby('time_of_day')['shares_to_buy'].sum().reset_index()
    aggregated_schedule = aggregated_schedule[aggregated_schedule['shares_to_buy'] > 0]

    plt.style.use('seaborn-v0_8-whitegrid')
    fig, ax = plt.subplots(figsize=(15, 7))
    ax.bar(aggregated_schedule['time_of_day'].astype(str), aggregated_schedule['shares_to_buy'], label='Aggregated Daily Schedule')
    
    ax.set_title('Optimal Execution Schedule (Aggregated by Time of Day)', fontsize=16)
    ax.set_xlabel('Time of Day (US/Eastern)', fontsize=12)
    ax.set_ylabel('Total Shares Executed at Time', fontsize=12)

    plt.xticks(rotation=45)
    ax.xaxis.set_major_locator(plt.MaxNLocator(15)) 
    
    ax.legend()
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    TOTAL_SHARES = 100000  # Example: 100,000 shares
    TICKER = 'FROG'      # Must be 'FROG', 'CRWV', or 'SOUN'
    LOT_SIZE = 500       # The size of each "block" of shares for the optimizer
    
    #  Load Pre-Computed Parameters from question 1
    params_file = f"{TICKER}_impact_parameters.csv"
    try:
        params_df = pd.read_csv(params_file)
        params_df = params_df.dropna() 
        params_df['gamma'] = 0.005 
    except FileNotFoundError:
        print(f"ERROR: The parameter file '{params_file}' was not found.")
        print("Please run the analysis script from Question 1 first to generate it.")
        exit()

    schedule, cost = run_execution_optimizer(params_df, TOTAL_SHARES, LOT_SIZE)

    if schedule is not None:
        print("\nOptimal Execution Schedule Found")
        schedule_df = pd.DataFrame({
            'timestamp': params_df['timestamp'],
            'shares_to_buy': schedule
        })
        print(schedule_df[schedule_df['shares_to_buy'] > 0])
        
        print("\n Summary ")
        print(f"Total Shares to Execute: {np.sum(schedule):,}")
        print(f"Estimated Total Slippage Cost: ${cost:,.2f}")
        print(f"Average Slippage Per Share: ${cost / np.sum(schedule):.4f}")
        
        plot_schedule(schedule_df)