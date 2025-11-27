import yfinance as yf

def get_current_stock_price(ticker_symbol: str) -> float | str:
    """
    Fetches the most recent stock price for a given ticker symbol.

    Args:
        ticker_symbol: The stock ticker symbol (e.g., "AAPL", "GOOG").

    Returns:
        The current stock price as a float, or an error message string if not found.
    """
    try:
        ticker = yf.Ticker(ticker_symbol)
        
        # Fetch the most recent data for the last day
        todays_data = ticker.history(period='1d')
        
        if todays_data.empty:
            return f"No data found for ticker {ticker_symbol}. It might be delisted or invalid."
            
        # Return the last closing price
        return todays_data['Close'].iloc[-1]
        
    except Exception as e:
        return f"An error occurred while fetching the price for {ticker_symbol}: {e}"

# --- Example Usage ---
if __name__ == '__main__':
    # Example with a valid ticker
    ticker = "SOXL"
    price = get_current_stock_price(ticker)
    if isinstance(price, float):
        print(f"The current price of {ticker} is: ${price:.2f}")
    else:
        print(price)

    # Example with an invalid ticker
    invalid_ticker = "NONEXISTENTTICKER"
    price_invalid = get_current_stock_price(invalid_ticker)
    print(f"Attempting to fetch '{invalid_ticker}': {price_invalid}")
