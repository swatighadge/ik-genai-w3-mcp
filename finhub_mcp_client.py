import gradio as gr
import matplotlib.pyplot as plt
import asyncio
from contextlib import AsyncExitStack
import json
import os

# --- Genkit and Google AI Imports ---
import genkit
import genkit.plugins.google_ai as google_ai
from genkit import flow
from dotenv import load_dotenv

# --- Correct Low-Level MCP Imports ---
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

# --- Initialize Genkit and Google AI Plugin ---
load_dotenv() # Load environment variables from .env file

genkit.init(
    plugins=[google_ai.init()],
    log_level="INFO",
    enable_tracing=True,
)

# --- Define the Genkit Flow for Summarization ---
@flow()
async def summarize_outlook(ticker: str, quote_data: dict, news_data: list, trends_data: list) -> str:
    """
    Uses a generative model to summarize the investment outlook.
    """
    # Create a detailed prompt for the model
    prompt = f"""
    You are a senior financial analyst providing a succinct investment outlook summary for the stock ticker: {ticker.upper()}.
    Synthesize the following data points into a clear, balanced, and insightful summary. Do not just list the data; interpret it.

    1.  **Current Price Action:**
        *   Current Price: ${quote_data.get('c', 'N/A'):.2f}
        *   Previous Close: ${quote_data.get('pc', 'N/A'):.2f}

    2.  **Analyst Recommendations:**
        *   Period: {trends_data[0].get('period', 'N/A')}
        *   Strong Buy: {trends_data[0].get('strongBuy', 0)}
        *   Buy: {trends_data[0].get('buy', 0)}
        *   Hold: {trends_data[0].get('hold', 0)}
        *   Sell: {trends_data[0].get('sell', 0)}
        *   Strong Sell: {trends_data[0].get('strongSell', 0)}

    3.  **Recent News Headlines:**
        {chr(10).join([f'- {article.get("headline", "No headline")}' for article in news_data])}

    **Your Task:**
    Based *only* on the data provided, generate a 3-4 sentence "Investment Outlook Summary". Cover the general analyst sentiment, mention the recent price movement, and incorporate the theme of the recent news.
    """

    # Define the generative model to use
    llm = google_ai.gemini_pro

    # Generate the content
    response = await llm.generate(prompt)
    return response.text()


# --- Analysis Logic ---
def analyze_news_sentiment(news_articles: list) -> (float, list):
    """Analyzes a list of news articles for sentiment."""
    if not news_articles or ("error" in news_articles[0] if news_articles else False):
        return 5.0, []

    positive_words = ['strong', 'growth', 'beat', 'exceeded', 'optimistic', 'record', 'upgrade', 'outperform']
    negative_words = ['missed', 'weak', 'decline', 'challenging', 'uncertainty', 'downgrade', 'underperform', 'risk']
    
    analyzed_headlines = []
    total_score = 0
    article_count = 0

    for article in news_articles:
        headline = article.get('headline', '').lower()
        if not headline:
            continue
        
        score = 0
        for p_word in positive_words:
            score += headline.count(p_word)
        for n_word in negative_words:
            score -= n_word.count(p_word)
        
        total_score += score
        article_count += 1
        
        sentiment_label = "NEUTRAL"
        if score > 0: sentiment_label = "POSITIVE"
        if score < 0: sentiment_label = "NEGATIVE"
        analyzed_headlines.append(f"[{sentiment_label}] {article.get('headline')}")

    if article_count == 0:
        return 5.0, []
        
    final_score = 5.0 + total_score
    return max(1, min(10, final_score)), analyzed_headlines

# --- Main Application Logic ---
async def analyze_and_plot(ticker: str):
    """
    Main function to call MCP tools, generate the summary, and create the plot.
    """
    if not ticker:
        return None, "Please enter a ticker symbol.", ""

    async with AsyncExitStack() as stack:
        try:
            server_params = StdioServerParameters(command=["python", "finhub_mcp_server.py"])
            read_pipe, write_pipe = await stack.enter_async_context(stdio_client(server_params))
            session = await stack.enter_async_context(ClientSession(read_pipe, write_pipe))
            await session.initialize()

            # Call all our MCP tools
            quote_res, news_res, trends_res = await asyncio.gather(
                session.call_tool(name="get_latest_quote", arguments={"ticker": ticker}),
                session.call_tool(name="get_company_news", arguments={"ticker": ticker}),
                session.call_tool(name="get_recommendation_trends", arguments={"ticker": ticker})
            )

            # Parse the JSON data from the tool responses
            quote_data = json.loads(quote_res.content[0].model_dump_json())
            news_data = [json.loads(item.model_dump_json()) for item in news_res.content]
            trends_data = [json.loads(item.model_dump_json()) for item in trends_res.content]

            if "error" in quote_data: return None, f"Error getting quote: {quote_data['error']}", ""
            if not trends_data: return None, "No recommendation data found.", ""

            # **NEW: Call the Genkit Flow to get the summary**
            summary = await summarize_outlook(ticker=ticker, quote_data=quote_data, news_data=news_data, trends_data=trends_data)

            # --- Visualization ---
            news_sentiment_score, analyzed_headlines = analyze_news_sentiment(news_data)
            
            trend = trends_data[0]
            labels = ['Strong Sell', 'Sell', 'Hold', 'Buy', 'Strong Buy']
            counts = [trend.get('strongSell', 0), trend.get('sell', 0), trend.get('hold', 0), trend.get('buy', 0), trend.get('strongBuy', 0)]
            colors = ['darkred', 'red', 'orange', 'skyblue', 'darkgreen']
            
            fig = plt.figure(figsize=(10, 8))
            gs = fig.add_gridspec(2, 1, height_ratios=[1, 1])

            ax1 = fig.add_subplot(gs[0])
            ax1.bar(labels, counts, color=colors)
            ax1.set_title(f'Analyst Recommendations for {ticker.upper()} ({trend.get("period")})')
            ax1.set_ylabel('Number of Analysts')
            
            current_price = quote_data.get('c', 0)
            prev_close = quote_data.get('pc', 0)
            change = current_price - prev_close
            change_percent = (change / prev_close) * 100 if prev_close != 0 else 0
            price_color = 'green' if change >= 0 else 'red'
            ax1.text(0.98, 0.95, f'Price: ${current_price:.2f}\nChg: ${change:+.2f} ({change_percent:+.2f}%)',
                     transform=ax1.transAxes, ha='right', va='top', color=price_color,
                     bbox=dict(boxstyle='round,pad=0.5', fc='white', alpha=0.7))

            ax2 = fig.add_subplot(gs[1])
            ax2.axis('off')
            ax2.set_title(f'Recent News (Sentiment Score: {news_sentiment_score:.1f}/10)')
            news_text = "\n\n".join(analyzed_headlines) if analyzed_headlines else "No recent news found."
            ax2.text(0, 0.9, news_text, ha='left', va='top', wrap=True, fontsize=9)

            plt.tight_layout(pad=3.0)
            return fig, f"Snapshot generated for {ticker.upper()}.", summary

        except Exception as e:
            return None, f"An application error occurred: {str(e)}", ""

# --- Gradio Interface ---
with gr.Blocks(css="footer {display: none !important}") as demo:
    gr.Markdown("# GenAI/MCP - Investment Advisor with Genkit Summary")
    gr.Markdown("Enter a stock ticker to get a snapshot of analyst recommendations, news sentiment, and an AI-generated investment outlook.")

    with gr.Row():
        ticker_input = gr.Textbox(label="Ticker Symbol", placeholder="e.g., GOOGL, MSFT")
        analyze_button = gr.Button("Analyze")
        
    status_output = gr.Textbox(label="Status", interactive=False)
    
    # **NEW: Added Markdown for the summary**
    summary_output = gr.Markdown(label="Investment Outlook Summary")
    
    plot_output = gr.Plot(label="Sentiment Snapshot")

    analyze_button.click(
        analyze_and_plot,
        inputs=[ticker_input],
        outputs=[plot_output, status_output, summary_output]
    )

if __name__ == "__main__":
    demo.launch()