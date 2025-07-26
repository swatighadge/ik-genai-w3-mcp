import finnhub
import os
from mcp.server.fastmcp import FastMCP 
from datetime import datetime, timedelta
import time
from dotenv import load_dotenv

load_dotenv()

# --- Configuration ---
# It's recommended to set your API key as an environment variable
# for better security practices.
# You can set it in your terminal like this:
# export FINNHUB_API_KEY='your_api_key'
FINNHUB_API_KEY = os.environ.get("FINNHUB_API_KEY")
if not FINNHUB_API_KEY:
    raise ValueError("Please set the FINNHUB_API_KEY environment variable.")

finnhub_client = finnhub.Client(api_key=FINNHUB_API_KEY)

# --- MCP Server Definition ---
mcp = FastMCP("finnhub-MCP-server")

@mcp.tool()
def get_latest_quote(ticker: str) -> dict:
    """
    Fetches the latest quote for a given stock ticker.
    This includes current price, previous close, high, low, etc.
    API: finnhub_client.quote()
    """
    try:
        return finnhub_client.quote(ticker)
    except Exception as e:
        return {"error": str(e)}

@mcp.tool()
def get_company_news(ticker: str) -> list:
    """
    Fetches recent company news for a given ticker.
    Returns the most recent 3 articles.
    API: finnhub_client.company_news()
    """
    try:
        end_date = datetime.now()
        start_date = end_date - timedelta(days=30) # Look for news in the last 30 days
        news = finnhub_client.company_news(ticker, _from=start_date.strftime("%Y-%m-%d"), to=end_date.strftime("%Y-%m-%d"))
        # Return the top 3 most recent articles
        return news[:3]
    except Exception as e:
        return [{"error": str(e)}]

@mcp.tool()
def get_recommendation_trends(ticker: str) -> list:
    """
    Fetches the latest analyst recommendation trends for a stock.
    API: finnhub_client.recommendation_trends()
    """
    try:
        # This API returns a list, usually with one element containing the trends.
        trends = finnhub_client.recommendation_trends(ticker)
        return trends
    except Exception as e:
        return [{"error": str(e)}]

@mcp.tool()
def get_stock_history(ticker: str) -> dict:
    """
    Fetches the last 7 days of stock price history for a given ticker.
    Returns a dictionary with dates and closing prices.
    """
    try:
        print("Getting stock history for ", str)
        end_date = datetime.now()
        start_date = end_date - timedelta(days=7)
        
        # Convert to timestamps
        end_timestamp = int(time.mktime(end_date.timetuple()))
        start_timestamp = int(time.mktime(start_date.timetuple()))

        history = finnhub_client.stock_candles(ticker, 'D', start_timestamp, end_timestamp)
        
        if history and 'c' in history:
            dates = [datetime.fromtimestamp(ts).strftime('%Y-%m-%d') for ts in history['t']]
            prices = history['c']
            return {"dates": dates, "prices": prices}
        else:
            return {"error": "No history found for this ticker."}
            
    except Exception as e:
        return {"error": str(e)}

@mcp.tool()
def get_earnings_reports(ticker: str) -> list:
    """
    Fetches the top 3 most recent earnings call transcripts for a given ticker.
    """
    try:
        print("Looking up finance reports for ", str)

        # Note: The free Finnhub plan has limitations on earnings calendar lookups.
        # This is a best-effort attempt.

        earnings = finnhub_client.earnings_calendar(_from="2025-01-01", to=datetime.now().strftime("%Y-%m-%d"), symbol=ticker, international=False)

        if earnings and 'earningsCalendar' in earnings:
            # Sort by date and get the top 3
            top_earnings = sorted(earnings['earningsCalendar'], key=lambda x: x['date'], reverse=True)[:3]
            return top_earnings
        else:
            return []
    except Exception as e:
        return [{"error": str(e)}]

# --- Run the server ---
if __name__ == "__main__":
    print("Strating finnhub MCP server")
    mcp.run(transport="stdio")