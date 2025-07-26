import gradio as gr
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import asyncio
from contextlib import AsyncExitStack
import json
import os
from datetime import datetime
import traceback

# --- GenAI and MCP Imports ---
import google.generativeai as genai
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

from dotenv import load_dotenv

load_dotenv()
# --- Configure Google AI ---
# Ensure the GEMINI_API_KEY environment variable is set
if not os.environ.get("GEMINI_API_KEY"):
    raise ValueError("Please set the GEMINI_API_KEY environment variable.")
genai.configure(api_key=os.environ.get("GEMINI_API_KEY"))

# --- Helper function for safe formatting ---
def format_price(price):
    """Safely formats a price value, returning 'N/A' if it's not a number."""
    if isinstance(price, (int, float)):
        return f"${price:.2f}"
    return "N/A"

# --- GenAI Summarization Logic (Now with robust formatting) ---
async def generate_investment_summary(ticker: str, quote_data: dict, history_data: dict, news_data: list, trends_data: dict) -> str:
    """Uses Google's Generative AI with robust data formatting."""
    model = genai.GenerativeModel('gemini-1.5-flash-latest')
    
    # Safely get historical prices to prevent IndexError
    start_price_str = format_price(history_data['prices'][0]) if history_data.get('prices') else "N/A"
    start_date_str = history_data['dates'][0] if history_data.get('dates') else "N/A"
    end_price_str = format_price(history_data['prices'][-1]) if history_data.get('prices') else "N/A"
    end_date_str = history_data['dates'][-1] if history_data.get('dates') else "N/A"

    prompt = f"""
    You are a financial analyst providing a concise investment outlook summary for the stock ticker: {ticker.upper()}.

    Based on the following real-time data, generate a summary in markdown format. Cover the price action, historical trend, analyst consensus, and key news drivers.

    **1. Current Price Data:**
    - Current Price: {format_price(quote_data.get('c'))}
    - Previous Close: {format_price(quote_data.get('pc'))}
    - Day's High: {format_price(quote_data.get('h'))}
    - Day's Low: {format_price(quote_data.get('l'))}

    **2. Historical 7-Day Trend:**
    - Start Price ({start_date_str}): {start_price_str}
    - End Price ({end_date_str}): {end_price_str}

    **3. Analyst Recommendation Trends:**
    - Period: {trends_data.get('period', 'N/A')}
    - Strong Buy: {trends_data.get('strongBuy', 0)}
    - Buy: {trends_data.get('buy', 0)}
    - Hold: {trends_data.get('hold', 0)}
    - Sell: {trends_data.get('sell', 0)}
    - Strong Sell: {trends_data.get('strongSell', 0)}

    **4. Recent News Headlines:**
    - {news_data[0].get('headline', 'N/A') if len(news_data) > 0 else 'N/A'}
    - {news_data[1].get('headline', 'N/A') if len(news_data) > 1 else 'N/A'}
    - {news_data[2].get('headline', 'N/A') if len(news_data) > 2 else 'N/A'}

    Provide your summary below:
    """

    try:
        response = await model.generate_content_async(prompt)
        return response.text
    except Exception as e:
        return f"Error generating summary: {str(e)}"

# --- Analysis Logic (Unchanged) ---
def analyze_news_sentiment(news_articles: list) -> (float, list):
    if not news_articles or ("error" in news_articles[0] if news_articles else False): return 5.0, []
    positive_words = ['strong', 'growth', 'beat', 'exceeded', 'optimistic', 'record', 'upgrade', 'outperform']
    negative_words = ['missed', 'weak', 'decline', 'challenging', 'uncertainty', 'downgrade', 'underperform', 'risk']
    analyzed_headlines = []; total_score = 0; article_count = 0
    for article in news_articles:
        headline = article.get('headline', '').lower()
        if not headline: continue
        score = sum(headline.count(p) for p in positive_words) - sum(headline.count(n) for n in negative_words)
        total_score += score; article_count += 1
        sentiment_label = "NEUTRAL" if score == 0 else "POSITIVE" if score > 0 else "NEGATIVE"
        analyzed_headlines.append(f"[{sentiment_label}] {article.get('headline')}")
    if article_count == 0: return 5.0, []
    final_score = 5.0 + total_score
    return max(1, min(10, final_score)), analyzed_headlines

# --- Main Application Logic (Updated with corrected JSON parsing) ---
async def analyze_and_plot(ticker: str):
    if not ticker:
        return None, "Please enter a ticker symbol.", "Enter a ticker to begin."
    async with AsyncExitStack() as stack:
        try:
            server_params = StdioServerParameters(command="python", args=["mcp_server.py"])
            read_pipe, write_pipe = await stack.enter_async_context(stdio_client(server_params))
            session = await stack.enter_async_context(ClientSession(read_pipe, write_pipe))
            await session.initialize()

            history_res, quote_res, news_res, trends_res = await asyncio.gather(
                session.call_tool(name="get_stock_history", arguments={"ticker": ticker}),
                session.call_tool(name="get_latest_quote", arguments={"ticker": ticker}),
                session.call_tool(name="get_company_news", arguments={"ticker": ticker}),
                session.call_tool(name="get_recommendation_trends", arguments={"ticker": ticker})
            )

            # === BUG FIX STARTS HERE ===
            # The JSON string is in the '.text' attribute of the 'TextContent' object.
            history_data = json.loads(history_res.content[0].text)
            quote_data = json.loads(quote_res.content[0].text)
            news_data = [json.loads(item.text) for item in news_res.content]
            trends_data = [json.loads(item.text) for item in trends_res.content]
            # === BUG FIX ENDS HERE ===

            if "error" in history_data: return None, f"History Error: {history_data['error']}", ""
            if "error" in quote_data: return None, f"Quote Error: {quote_data['error']}", ""
            if not trends_data or ("error" in trends_data[0] if trends_data else False): return None, "No recommendation data.", ""
            if not news_data or ("error" in news_data[0] if news_data else False): return None, "No news data.", ""

            summary_text = await generate_investment_summary(ticker, quote_data, history_data, news_data, trends_data[0])
            news_sentiment_score, analyzed_headlines = analyze_news_sentiment(news_data)
            
            fig = plt.figure(figsize=(12, 16))
            gs = fig.add_gridspec(3, 1, height_ratios=[2, 2, 1])

            # Subplot 1: Historical Price Trend
            ax1 = fig.add_subplot(gs[0])
            dates = [datetime.strptime(d, "%Y-%m-%d") for d in history_data['dates']]
            ax1.plot(dates, history_data['prices'], 'o-', label='Closing Price', color='royalblue')
            ax1.set_title(f'7-Day Price History for {ticker.upper()}', fontsize=14)
            ax1.set_ylabel('Price (USD)')
            ax1.grid(True, linestyle='--', alpha=0.6)
            ax1.xaxis.set_major_formatter(mdates.DateFormatter('%b %d'))
            ax1.legend()
            
            # Subplot 2: Analyst Recommendations
            ax2 = fig.add_subplot(gs[1])
            trend = trends_data[0]
            labels = ['Strong Sell', 'Sell', 'Hold', 'Buy', 'Strong Buy']
            counts = [trend.get(k, 0) for k in ['strongSell', 'sell', 'hold', 'buy', 'strongBuy']]
            colors = ['darkred', 'red', 'orange', 'skyblue', 'darkgreen']
            ax2.bar(labels, counts, color=colors)
            ax2.set_title(f'Analyst Recommendations ({trend.get("period")})', fontsize=14)
            ax2.set_ylabel('Number of Analysts')
            
            current_price = quote_data.get('c', 0)
            change = quote_data.get('d', 0)
            change_percent = quote_data.get('dp', 0)
            price_color = 'green' if (isinstance(change, (int, float)) and change >= 0) else 'red'
            if isinstance(current_price, (int, float)) and isinstance(change, (int, float)) and isinstance(change_percent, (int, float)):
                 ax2.text(0.98, 0.95, f'Price: ${current_price:.2f}\nChg: ${change:+.2f} ({change_percent:+.2f}%)',
                     transform=ax2.transAxes, ha='right', va='top', color=price_color,
                     bbox=dict(boxstyle='round,pad=0.5', fc='white', alpha=0.8))

            # Subplot 3: News Headlines
            ax3 = fig.add_subplot(gs[2])
            ax3.axis('off')
            ax3.set_title(f'Recent News (Sentiment Score: {news_sentiment_score:.1f}/10)', fontsize=14)
            news_text = "\n".join(analyzed_headlines) if analyzed_headlines else "No recent news found."
            ax3.text(0, 0.9, news_text, ha='left', va='top', wrap=True, fontsize=10)

            plt.tight_layout(pad=3.0)
            return fig, f"Analysis complete for {ticker.upper()}.", summary_text

        except Exception as e:
            traceback.print_exc()
            return None, f"An application error occurred: {str(e)}", ""

# --- Gradio Interface (Unchanged) ---
with gr.Blocks(css="footer {display: none !important}") as demo:
    gr.Markdown("# GenAI Investment Advisor with MCP")
    gr.Markdown("Enter a stock ticker to get a complete visual analysis and an AI-generated investment outlook.")
    with gr.Row():
        ticker_input = gr.Textbox(label="Ticker Symbol", placeholder="e.g., NVDA, MSFT, GOOG")
        analyze_button = gr.Button("Analyze")
    status_output = gr.Textbox(label="Status", interactive=False)
    plot_output = gr.Plot(label="Full Ticker Snapshot")
    summary_output = gr.Markdown(label="AI Investment Outlook")
    analyze_button.click(
        analyze_and_plot,
        inputs=[ticker_input],
        outputs=[plot_output, status_output, summary_output]
    )

if __name__ == "__main__":
    demo.launch()