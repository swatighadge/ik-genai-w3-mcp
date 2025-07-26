MCP Server for Week3 of GenAI 

Goto: https://finnhub.io/ and sign-up for a free API key. This API key will be used in the program to access finnhub APIs.

Goto https://github.com/manupatet/ik-genai-w3-mcp
Fork this repository

Goto clone->codespaces. This will open the web IDE where we can make the subsequent changes.



In the IDE that opens, create a new .env file (by right clicking on the files pane) and add the following lines (with correct keys):
FINNHUB_API_KEY=xxxxx
GEMINI_API_KEY=xxxxxx

Open terminal and type these commands:
uv venv
source .venv/bin/activate

You’re now ready to run the MCP client and servers.
Run client using: uv run finhub_mcp_client.py and click “Open in browser”.

On the new browser window that opens, use the gradio UI to communicate with your App.
MCP inspector tool: npx @modelcontextprotocol/inspector
