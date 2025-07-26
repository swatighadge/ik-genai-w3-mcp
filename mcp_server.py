import finnhub
import os
from mcp.server.fastmcp import FastMCP 
from datetime import datetime, timedelta
from dotenv import load_dotenv
import yfinance as yf

load_dotenv()

# --- Configuration ---
FINNHUB_API_KEY = os.environ.get("FINNHUB_API_KEY")
if not FINNHUB_API_KEY:
    raise ValueError("Please set the FINNHUB_API_KEY environment variable.")

finnhub_client = finnhub.Client(api_key=FINNHUB_API_KEY)

# --- MCP Server Definition ---
mcp = FastMCP("finnhub-MCP-server")

# --- THIS TOOL IS NOW POWERED BY YFINANCE ---
@mcp.tool()
def get_stock_history(ticker: str) -> dict:
    """
    Fetches the last 7 days of stock price history for a given ticker using yfinance.
    Returns a dictionary with dates and closing prices.
    """
    try:
        stock = yf.Ticker(ticker)
        # yfinance makes fetching recent history very simple
        history = stock.history(period="7d")

        if history.empty:
            return {"error": f"No history found for ticker '{ticker}'. It may be invalid."}

        # The output format (the "contract") MUST remain the same for the client.
        history.reset_index(inplace=True)
        dates = history['Date'].dt.strftime('%Y-%m-%d').tolist()
        prices = history['Close'].tolist()
        
        return {"dates": dates, "prices": prices}

    except Exception as e:
        return {"error": str(e)}

# --- THESE TOOLS STILL USE FINNHUB ---
@mcp.tool()
def get_latest_quote(ticker: str) -> dict:
    """
    Fetches the latest quote for a given stock ticker.
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
    API: finnhub_client.company_news()
    """
    try:
        end_date = datetime.now()
        start_date = end_date - timedelta(days=30)
        news = finnhub_client.company_news(ticker, _from=start_date.strftime("%Y-%m-%d"), to=end_date.strftime("%Y-%m-%d"))
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
        trends = finnhub_client.recommendation_trends(ticker)
        return trends
    except Exception as e:
        return [{"error": str(e)}]

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
    print("Starting Finnhub/yfinance hybrid MCP server")
    mcp.run(transport="stdio")