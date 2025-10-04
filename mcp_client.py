import gradio as gr
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import asyncio
from contextlib import AsyncExitStack
import json
import os
from datetime import datetime
import traceback
import inspect
from typing import List

# --- GenAI and MCP Imports ---
import google.generativeai as genai
from google.generativeai.types import GenerationConfig
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client
from mcp.types import CallToolResult

from dotenv import load_dotenv

load_dotenv()
# --- Configure Google AI ---
if not os.environ.get("GEMINI_API_KEY"):
    raise ValueError("Please set the GEMINI_API_KEY environment variable.")
genai.configure(api_key=os.environ.get("GEMINI_API_KEY"))

# --- Helper function for JSON parsing ---
def parse_json_response(res: CallToolResult) -> str:
    """Parse a response object with .content (list of items having .text)
    and return a stringified JSON (always str)."""
    try:
        content = res.content
        if not content:
            return json.dumps({"error": "No content received"})

        if len(content) == 1:
            data = json.loads(content[0].text)
        else:
            data = [json.loads(item.text) for item in content]

        return json.dumps(data)
    except json.JSONDecodeError as e:
        if content and hasattr(content[0], 'text'):
            return json.dumps({"error": f"JSON decode error: {e}", "raw_text": content[0].text})
        return json.dumps({"error": f"JSON decode error: {e}"})
    except Exception as e:
        return json.dumps({"error": f"An unexpected error occurred during parsing: {e}"})


# --- Local Report-Building Tools ---
# These tools DO NOT fetch data. They render it.

def plot_historical_price_chart(data_json: str) -> plt.Figure:
    try:
        history_data = json.loads(data_json)
        if "error" in history_data or not history_data.get('dates') or not history_data.get('prices'):
            return None

        fig, ax = plt.subplots(figsize=(10, 5))
        dates = [datetime.strptime(d, "%Y-%m-%d") for d in history_data['dates']]
        ax.plot(dates, history_data['prices'], 'o-', label='Closing Price', color='royalblue', markersize=6)
        ax.set_title('7-Day Price History', fontsize=14)
        ax.set_ylabel('Price (USD)')
        ax.grid(True, linestyle='--', alpha=0.6)
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%b %d'))
        ax.legend()
        fig.autofmt_xdate()
        plt.tight_layout(pad=2.0)
        return fig
    except Exception as e:
        return None

def plot_analyst_recommendations_chart(data_json: str) -> plt.Figure:
    """
    Accepts analyst recommendation data in JSON format and plots a bar chart.
    The JSON should contain keys like 'strongBuy', 'buy', 'hold', 'sell', 'strongSell'.
    """
    try:
        trends_data = json.loads(data_json)
        trend = trends_data[0] if isinstance(trends_data, list) and trends_data else trends_data
        if "error" in trend:
             raise ValueError("Invalid or error data provided for recommendations chart.")

        fig, ax = plt.subplots(figsize=(10, 5))
        labels = ['Strong Sell', 'Sell', 'Hold', 'Buy', 'Strong Buy']
        counts = [trend.get(k, 0) for k in ['strongSell', 'sell', 'hold', 'buy', 'strongBuy']]
        colors = ['darkred', 'red', 'orange', 'skyblue', 'darkgreen']
        ax.bar(labels, counts, color=colors)
        ax.set_title(f'Analyst Recommendations ({trend.get("period", "N/A")})', fontsize=14)
        ax.set_ylabel('Number of Analysts')
        plt.tight_layout(pad=2.0)
        return fig
    except Exception:
        fig, ax = plt.subplots(figsize=(10, 5))
        ax.text(0.5, 0.5, 'Could not generate recommendations plot.\nData may be missing or invalid.',
                ha='center', va='center', fontsize=12, color='red')
        ax.axis('off')
        return fig


async def generate_financial_summary(full_context_json: str) -> str:
    """
    Accepts the full data context (as a JSON string) and uses a generative model
    to create a "Summary" and "Overall Outlook" for the report.
    The level of detail will depend on the richness of the provided context.
    """
    model = genai.GenerativeModel('gemini-2.5-flash-lite-preview-06-17')
    prompt = f"""
    You are a financial analyst AI. Based *only* on the data provided in the following JSON context,
    generate a concise financial report.

    The report should have two sections:
    1.  **Summary:** A brief overview of the current stock situation based on the data.
    2.  **Overall Outlook:** A forward-looking statement. If news data is available,
        base the outlook on the sentiment and headlines. Otherwise, base it on price trends
        and analyst recommendations. Be neutral if the data is conflicting or sparse.

    Do not invent any information. If a piece of data (like news or quotes) is missing,
    simply state that it was not available and adapt your analysis accordingly.
    Format the output in markdown.

    **Full Data Context:**
    ```json
    {full_context_json}
    ```
    """
    try:
        response = await model.generate_content_async(prompt)
        return response.text
    except Exception as e:
        return f"### Error Generating Summary\nAn error occurred while communicating with the AI model: {str(e)}"

async def analyze_and_plot(ticker: str):
    if not ticker:
        return None, "Please enter a ticker symbol.", "Enter a ticker to begin."

    full_context = {}
    async with AsyncExitStack() as stack:
        try:
            # 1. AGGREGATE 
            server_params = StdioServerParameters(command="python", args=["mcp_server.py"])
            read_pipe, write_pipe = await stack.enter_async_context(stdio_client(server_params))
            session = await stack.enter_async_context(ClientSession(read_pipe, write_pipe))
            await session.initialize()

            available_tools = await session.list_tools()
            tool_names = [tool.name for tool in available_tools.tools]

            tasks = [session.call_tool(name=name, arguments={"ticker": ticker}) for name in tool_names]
            results = await asyncio.gather(*tasks, return_exceptions=True)

            for name, res in zip(tool_names, results):
                if isinstance(res, Exception):
                    full_context[name] = json.dumps({"error": f"Failed to call tool: {res}"})
                else:
                    full_context[name] = parse_json_response(res)

            full_context_json_str = json.dumps(full_context, indent=2)

            # 2. DEFINE AND INVOKE AGENT WITH LOCAL TOOLS
            local_tools = [
                plot_historical_price_chart,
                plot_analyst_recommendations_chart,
                generate_financial_summary,
            ]
            agent = genai.GenerativeModel(
                'gemini-2.5-flash-lite-preview-06-17',
                tools=local_tools,
                generation_config=GenerationConfig(temperature=0.0)
            )

            # 3. COMPOSE AGENT PROMPT
            prompt = f"""
            You are an expert financial analyst. Your job is to build a report by calling the provided tools based on the data context you receive.
            Do not make up data. Only call a tool if the required data is present and valid in the context.

            Here is the complete data context for the analysis of ticker '{ticker.upper()}':
            ```json
            {full_context_json_str}
            ```
            Your task is to:
            1. Call `plot_historical_price_chart` if 'get_stock_history' data exists and is not an error.
            2. Call `plot_analyst_recommendations_chart` if 'get_recommendation_trends' data exists and is not an error.
            3. Finally, after analyzing all available data, call `generate_financial_summary` with the complete data context to create the textual report.
            """

            # 4. EXECUTE AGENT AND RENDER RESULTS
            response = await agent.generate_content_async(prompt)
            # 4. EXECUTE AGENT AND RENDER RESULTS
            tool_calls = getattr(response.candidates[0].content, "parts", []) or []

            tool_exec_map = {
                "plot_historical_price_chart": plot_historical_price_chart,
                "plot_analyst_recommendations_chart": plot_analyst_recommendations_chart,
                "generate_financial_summary": generate_financial_summary
            }

            sync_results = []
            async_tasks = []

            # --- Safety: if Gemini fails to call tools, fallback manually ---
            called_tool_names = set()

            for call in tool_calls:
                if getattr(call, "function_call", None):
                    fn = call.function_call
                    called_tool_names.add(fn.name)
                    if fn.name in tool_exec_map:
                        tool_func = tool_exec_map[fn.name]
                        if inspect.iscoroutinefunction(tool_func):
                            async_tasks.append(tool_func(**fn.args))
                        else:
                            sync_results.append(tool_func(**fn.args))

            # --- Fallback: ensure we *always* run summary even if Gemini didn't call it ---
            if "generate_financial_summary" not in called_tool_names:
                async_tasks.append(generate_financial_summary(full_context_json_str))

            async_results = []
            if async_tasks:
                async_results = await asyncio.gather(*async_tasks)

            tool_results = sync_results + async_results

            # --- Separate plots vs summaries ---
            figures = [res for res in tool_results if isinstance(res, plt.Figure) and res is not None]
            summaries = [res for res in tool_results if isinstance(res, str) and res.strip()]

            # --- Handle missing figures gracefully ---
            final_plot = None
            if len(figures) > 1:
                combined_fig, axes = plt.subplots(len(figures), 1, figsize=(10, 5 * len(figures)))
                if len(figures) == 1:
                    axes = [axes]
                for i, fig in enumerate(figures):
                    ax_old = fig.get_axes()[0]
                    ax_new = axes[i]
                    ax_new.set_title(ax_old.get_title())
                    ax_new.set_ylabel(ax_old.get_ylabel())
                    for line in ax_old.get_lines():
                        ax_new.plot(line.get_xdata(), line.get_ydata(),
                                    linestyle=line.get_linestyle(),
                                    linewidth=line.get_linewidth(),
                                    color=line.get_color(),
                                    marker=line.get_marker(),
                                    markersize=line.get_markersize(),
                                    label=line.get_label())
                    if ax_old.patches:
                        for bar in ax_old.patches:
                            ax_new.add_patch(
                                plt.Rectangle(
                                    (bar.get_x(), bar.get_y()), bar.get_width(), bar.get_height(),
                                    facecolor=bar.get_facecolor(), edgecolor=bar.get_edgecolor()
                                )
                            )
                        ax_new.set_xlim(ax_old.get_xlim())
                        ax_new.set_ylim(ax_old.get_ylim())
                        ax_new.set_xticks(ax_old.get_xticks())
                        ax_new.set_xticklabels([label.get_text() for label in ax_old.get_xticklabels()])
                    plt.close(fig)
                combined_fig.tight_layout(pad=3.0)
                final_plot = combined_fig
            elif len(figures) == 1:
                final_plot = figures[0]

            # --- Always produce a summary even if no data/plots exist ---
            final_summary = (
                "\n\n".join(summaries)
                if summaries
                else "### No data available\nCould not fetch historical or trend data for this ticker, but the app is functioning correctly."
            )

            return final_plot, f"Analysis complete for {ticker.upper()}.", final_summary

        except Exception as e:
            traceback.print_exc()
            return None, f"An application error occurred: {str(e)}", ""

# --- Gradio Interface (Unchanged) ---
with gr.Blocks(css="footer {display: none !important}") as demo:
    gr.Markdown("# My GenAI Investment Advisor")
    gr.Markdown("Enter a stock ticker. The app will discover available data APIs, fetch all data, and an AI agent will dynamically build a report based on what it finds.")
    with gr.Row():
        ticker_input = gr.Textbox(label="Ticker Symbol", placeholder="e.g., NVDA, MSFT, GOOG")
        analyze_button = gr.Button("Analyze")
    status_output = gr.Textbox(label="Status", interactive=False)
    plot_output = gr.Plot(label="Visual Report")
    summary_output = gr.Markdown(label="Stock Summary & Outlook")
    analyze_button.click(
        analyze_and_plot,
        inputs=[ticker_input],
        outputs=[plot_output, status_output, summary_output]
    )

if __name__ == "__main__":
    demo.launch()