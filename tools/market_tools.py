import yfinance as yf
from langchain_core.tools import tool

@tool
def get_stock_price(ticker: str):
    """
    Retrieves the current stock price and basic info for a given ticker symbol.
    Args:
        ticker: The stock ticker symbol (e.g., AAPL, MSFT).
    """
    try:
        stock = yf.Ticker(ticker)
        info = stock.info
        current_price = info.get('currentPrice') or info.get('regularMarketPrice')
        currency = info.get('currency', 'USD')
        return f"The current price of {ticker} is {current_price} {currency}."
    except Exception as e:
        return f"Error fetching stock price for {ticker}: {e}"
