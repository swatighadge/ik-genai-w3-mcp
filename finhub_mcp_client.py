import gradio as gr
import matplotlib.pyplot as plt
import asyncio
from contextlib import AsyncExitStack
import json # We need the standard json library for parsing

# --- Correct Low-Level MCP Imports ---
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

# --- Analysis Logic ---
def analyze_news_sentiment(news_articles: list) -> (float, list):
    """Analyzes a list of news articles for sentiment."""
    if not news_articles or "error" in news_articles[0]:
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
            score -= headline.count(n_word)
        
        total_score += score
        article_count += 1
        
        sentiment_label = "NEUTRAL"
        if score > 0: sentiment_label = "POSITIVE"
        if score < 0: sentiment_label = "NEGATIVE"
        analyzed_headlines.append(f"[{sentiment_label}] {article.get('headline')}")

    # Average score, scaled to 1-10 range
    if article_count == 0:
        return 5.0, []
        
    final_score = 5.0 + total_score
    return max(1, min(10, final_score)), analyzed_headlines

# --- Main Application Logic ---

async def analyze_and_plot(ticker: str):
    """
    Main function to call MCP tools and generate the Sentiment Snapshot.
    """
    if not ticker:
        return None, "Please enter a ticker symbol."

    async with AsyncExitStack() as stack:
        try:
            server_params = StdioServerParameters(command="uv", args=["run","finhub_mcp_server.py"])
            read_pipe, write_pipe = await stack.enter_async_context(stdio_client(server_params))
            session = await stack.enter_async_context(ClientSession(read_pipe, write_pipe))
            await session.initialize()

            # --- Call all our new tools ---
            quote_res = await session.call_tool(name="get_latest_quote", arguments={"ticker": ticker})
            news_res = await session.call_tool(name="get_company_news", arguments={"ticker": ticker})
            trends_res = await session.call_tool(name="get_recommendation_trends", arguments={"ticker": ticker})

            # --- *** FINAL, CORRECTED DATA EXTRACTION *** ---
            #
            # Adhere to the Pydantic V2 deprecation warning.
            # 1. Use .model_dump_json() to get a JSON string.
            # 2. Use json.loads() to parse the string into a Python dictionary.
            #
            quote_data = json.loads(quote_res.content[0].model_dump_json())
            news_data = [json.loads(item.model_dump_json()) for item in news_res.content]
            trends_data = [json.loads(item.model_dump_json()) for item in trends_res.content]

            # Quote returns a single object, so we access the first content item.
            if "error" in quote_data: return None, f"Error getting quote: {quote_data['error']}"
            if not trends_data: return None, "No recommendation data found."

            news_sentiment_score, analyzed_headlines = analyze_news_sentiment(news_data)
            
            trend = trends_data[0]
            labels = ['Strong Sell', 'Sell', 'Hold', 'Buy', 'Strong Buy']
            counts = [trend.get('strongSell', 0), trend.get('sell', 0), trend.get('hold', 0), trend.get('buy', 0), trend.get('strongBuy', 0)]
            colors = ['darkred', 'red', 'orange', 'skyblue', 'darkgreen']
            
            # Create the Plot
            fig = plt.figure(figsize=(10, 8))
            gs = fig.add_gridspec(2, 1, height_ratios=[1, 1])

            # Subplot 1: Recommendation Bars
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

            # Subplot 2: News Headlines
            ax2 = fig.add_subplot(gs[1])
            ax2.axis('off')
            ax2.set_title(f'Recent News (Sentiment Score: {news_sentiment_score:.1f}/10)')
            news_text = "\n\n".join(analyzed_headlines) if analyzed_headlines else "No recent news found."
            ax2.text(0, 0.9, news_text, ha='left', va='top', wrap=True, fontsize=9)

            plt.tight_layout(pad=3.0)
            return fig, f"Snapshot generated for {ticker.upper()}."

        except Exception as e:
            return None, f"An application error occurred: {str(e)}"

# --- Gradio Interface ---
with gr.Blocks(css="footer {display: none !important}") as demo:
    gr.Markdown("# GenAI/week3/MCP - Investment Advisor")
    gr.Markdown("Enter a stock ticker to get a snapshot of analyst recommendations and news sentiment.")

    with gr.Row():
        ticker_input = gr.Textbox(label="Ticker Symbol", placeholder="e.g., AAPL, TSLA")
        analyze_button = gr.Button("Analyze")
        
    status_output = gr.Textbox(label="Status", interactive=False)
    plot_output = gr.Plot(label="Sentiment Snapshot")

    analyze_button.click(
        analyze_and_plot,
        inputs=[ticker_input],
        outputs=[plot_output, status_output]
    )

if __name__ == "__main__":
    demo.launch()