# MCP Server for Week3 of GenAI 

Fork this repository

Goto: https://finnhub.io/ and sign-up for a free API key. This API key will be used in the program to access finnhub APIs.

Goto `clone->codespaces`. This will open the web IDE where we can make the subsequent changes.

In the IDE that opens, create a new `.env` file (by right clicking on the files pane) and add the following lines (with correct keys):
```
FINNHUB_API_KEY=xxxxx
GEMINI_API_KEY=xxxxxx
```

Open terminal and type these commands:
- `uv venv`
- `source .venv/bin/activate`

You’re now ready to run the MCP client and servers.
Run client using: `uv run mcp_client.py` and click “Open in browser”.

On the new browser window that opens, use the gradio UI to communicate with your App.
MCP inspector tool: `npx @modelcontextprotocol/inspector`

###Appendix
Free APIs on finnhub:
```
print(finnhub_client.company_basic_financials('AAPL', 'all'))
print(finnhub_client.company_earnings('TSLA', limit=5))
print(finnhub_client.company_news('AAPL', _from="2020-06-01", to="2020-06-10"))
print(finnhub_client.company_peers('AAPL'))
print(finnhub_client.company_profile2(symbol='AAPL'))
print(finnhub_client.country())
print(finnhub_client.crypto_exchanges())
print(finnhub_client.crypto_symbols('BINANCE'))
print(finnhub_client.filings(symbol='AAPL', _from="2020-01-01", to="2020-06-11"))
print(finnhub_client.financials('AAPL', 'bs', 'annual'))
print(finnhub_client.financials_reported(symbol='AAPL', freq='annual'))
print(finnhub_client.forex_exchanges())
print(finnhub_client.forex_symbols('OANDA'))
print(finnhub_client.general_news('forex', min_id=0))
print(finnhub_client.ipo_calendar(_from="2020-05-01", to="2020-06-01"))
print(finnhub_client.quote('AAPL'))
print(finnhub_client.recommendation_trends('AAPL'))
print(finnhub_client.stock_symbols('US')[0:5])
print(finnhub_client.earnings_calendar(_from="2020-06-10", to="2020-06-30", symbol="", international=False))
print(finnhub_client.covid19())
print(finnhub_client.fda_calendar())
print(finnhub_client.symbol_lookup('apple'))
print(finnhub_client.stock_insider_transactions('AAPL', '2021-01-01', '2021-03-01'))
print(finnhub_client.stock_visa_application("AAPL", "2021-01-01", "2022-06-15"))
print(finnhub_client.stock_insider_sentiment('AAPL', '2021-01-01', '2022-03-01'))
print(finnhub_client.stock_lobbying("AAPL", "2021-01-01", "2022-06-15"))
print(finnhub_client.stock_usa_spending("LMT", "2021-01-01", "2022-06-15"))
print(finnhub_client.market_holiday(exchange='US'))
print(finnhub_client.market_status(exchange='US'))
```