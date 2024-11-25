import pandas as pd
from binance.client import Client
from datetime import datetime
import time
import os
import logging
from collections import defaultdict, deque

# *** Step 1: Configure Logging ***
logging.basicConfig(
    filename='crypto_tax_calculator.log',
    level=logging.INFO,
    format='%(asctime)s %(levelname)s:%(message)s'
)
console = logging.StreamHandler()
console.setLevel(logging.INFO)
formatter = logging.Formatter('%(asctime)s %(levelname)s:%(message)s')
console.setFormatter(formatter)
logging.getLogger('').addHandler(console)

# *** Step 2: Define Folder Path and CSV File ***
folder_path = r"C:\Users\nikla\OneDrive\Dokumente\crypto\crypto_steuern"
csv_file = "binance_2017_2018.csv"  # Update the file name if necessary
csv_path = os.path.join(folder_path, csv_file)

# *** Step 3: Initialize Binance Client ***
client = Client(api_key='', api_secret='')  # API keys are not required for public data

# *** Step 4: Function to Fetch Current USDT Prices from Binance ***
def get_current_usdt_prices_binance():
    """
    Fetches the current USDT prices of all available coins from Binance.
    
    Returns:
        dict: Mapping of Coin symbols to their current USDT prices.
    """
    try:
        tickers = client.get_all_tickers()
        usdt_prices = {}
        for ticker in tickers:
            symbol = ticker['symbol']
            price = float(ticker['price'])
            if symbol.endswith('USDT'):
                coin = symbol[:-4]  # Remove 'USDT' from the symbol
                usdt_prices[coin.upper()] = price
        logging.info("Fetched current USDT prices from Binance.")
        return usdt_prices
    except Exception as e:
        logging.error(f"Error fetching USDT prices from Binance: {e}")
        return {}

# *** Step 5: Load and Prepare Data ***
def load_and_prepare_data(csv_path):
    """Loads the CSV file and sorts transactions by UTC_Time."""
    try:
        df = pd.read_csv(csv_path)
        df['UTC_Time'] = pd.to_datetime(df['UTC_Time'])
        df = df.sort_values('UTC_Time').reset_index(drop=True)
        logging.info("Loaded data and sorted by UTC_Time.")
        logging.info(f"First few rows of the data:\n{df.head()}")
        return df
    except Exception as e:
        logging.error(f"Error loading CSV file: {e}")
        return pd.DataFrame()

# *** Step 6: Group Transactions ***
def group_transactions(df):
    """
    Groups transactions by User_ID and UTC_Time.
    
    Args:
        df (DataFrame): Pandas DataFrame with transactions.
    
    Returns:
        list of DataFrame: List of grouped transactions.
    """
    grouped = df.groupby(['User_ID', 'UTC_Time'])
    return [group for name, group in grouped]

# *** Step 7: Calculate Profits ***
def calculate_profits(grouped_transactions, usdt_prices):
    """
    Calculates profits based on current USDT prices.
    
    Args:
        grouped_transactions (list of DataFrame): Grouped transactions.
        usdt_prices (dict): Mapping of Coin symbols to USDT prices.
    
    Returns:
        list of dict: List of positions with profit details.
    """
    positions = []
    for group in grouped_transactions:
        # Identify Sell, Buy, and Fee Operations
        sells = group[group['Operation'].str.lower() == 'sell']
        buys = group[group['Operation'].str.lower() == 'buy']
        fees = group[group['Operation'].str.lower() == 'fee']
        
        if sells.empty or buys.empty:
            continue  # Only process groups with at least one Sell and one Buy
        
        # Assume each Sell corresponds to a Buy in the same group
        for _, sell in sells.iterrows():
            for _, buy in buys.iterrows():
                sell_coin = sell['Coin'].upper()
                sell_amount = abs(float(sell['Change']))
                buy_coin = buy['Coin'].upper()
                buy_amount = float(buy['Change'])
                
                # Fetch current prices
                sell_price = usdt_prices.get(sell_coin, None)
                buy_price = usdt_prices.get(buy_coin, None)
                
                if sell_price is None:
                    logging.error(f"No current USDT price for sold coin '{sell_coin}'. Skipping...")
                    continue
                if buy_price is None:
                    logging.error(f"No current USDT price for bought coin '{buy_coin}'. Skipping...")
                    continue
                
                # Calculate costs and values
                cost_usd = sell_amount * sell_price
                value_usd = buy_amount * buy_price
                
                # Calculate fees in USD
                fee_usd = 0.0
                fee_entries = fees[(fees['Coin'].str.upper() == sell_coin) | (fees['Coin'].str.upper() == buy_coin)]
                for _, fee in fee_entries.iterrows():
                    fee_coin = fee['Coin'].upper()
                    fee_change = abs(float(fee['Change']))
                    fee_price = usdt_prices.get(fee_coin, None)
                    if fee_price:
                        fee_usd += fee_change * fee_price
                    else:
                        logging.error(f"No current USDT price for fee coin '{fee_coin}'. Fee ignored.")
                
                # Calculate profit
                profit_usd = value_usd - cost_usd - fee_usd
                
                position = {
                    'Sell_Coin': sell_coin,
                    'Sell_Amount': sell_amount,
                    'Sell_Price_USD': sell_price,
                    'Buy_Coin': buy_coin,
                    'Buy_Amount': buy_amount,
                    'Buy_Price_USD': buy_price,
                    'Cost_USD': cost_usd,
                    'Value_USD': value_usd,
                    'Fee_USD': fee_usd,
                    'Profit_USD': profit_usd
                }
                
                positions.append(position)
                
                logging.info(f"Created Position: {position}")
                
                # Optional: Add more logic here, e.g., FIFO tracking
                
                # Short pause to respect rate limits
                time.sleep(0.05)  # 50 milliseconds pause
    return positions

# *** Step 8: Analyze and Output Results ***
def analyze_and_output_results(positions, folder_path, year):
    """Analyzes positions and saves the results."""
    if not positions:
        logging.info("No positions found for analysis.")
        return
    
    profit_df = pd.DataFrame(positions)
    
    # Calculate total profit/loss
    total_profit_usd = profit_df['Profit_USD'].sum()
    usd_to_eur_rate = 0.90  # Example rate, adjust or integrate an API for dynamic conversion
    total_profit_eur = total_profit_usd * usd_to_eur_rate
    
    # Log total profit
    logging.info(f"Total Profit/Loss for {year}: {total_profit_usd:.2f} USD ({total_profit_eur:.2f} EUR)")
    
    # Print total profit to console
    print(f"\nTotal Profit/Loss for {year}: {total_profit_usd:.2f} USD ({total_profit_eur:.2f} EUR)\n")
    
    # Detailed profits/losses per position
    logging.info("\nDetailed Profits/Losses per Position:")
    logging.info(f"{profit_df}")
    
    # Save detailed profits/losses to a CSV file
    profit_details_path = os.path.join(folder_path, f'profit_details_{year}.csv')
    try:
        profit_df.to_csv(profit_details_path, index=False)
        logging.info(f"Detailed profits/losses saved to '{profit_details_path}'.")
    except Exception as e:
        logging.error(f"Error saving results to CSV: {e}")

# *** Step 9: Main Function ***
def main():
    # Load and prepare data
    df = load_and_prepare_data(csv_path)
    
    if df.empty:
        logging.error("No data to process. Exiting script.")
        return
    
    # Filter data for the year 2017
    df_2017 = df[df['UTC_Time'].dt.year == 2017].reset_index(drop=True)
    logging.info("Filtered data for the year 2017.")
    
    if df_2017.empty:
        logging.error("No data found for the year 2017. Exiting script.")
        return
    
    # Fetch current USDT prices from Binance
    usdt_prices = get_current_usdt_prices_binance()
    
    if not usdt_prices:
        logging.error("No USDT prices fetched. Exiting script.")
        return
    
    # Group transactions by User_ID and UTC_Time
    grouped_transactions = group_transactions(df_2017)
    logging.info(f"Found {len(grouped_transactions)} transaction groups.")
    
    # Calculate profits
    positions = calculate_profits(grouped_transactions, usdt_prices)
    
    # Analyze and output results
    analyze_and_output_results(positions, folder_path, year=2017)

if __name__ == "__main__":
    main()
