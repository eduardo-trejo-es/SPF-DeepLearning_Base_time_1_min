import yfinance as yf
data = yf.download("TWTR", start="2022-06-14", end="2022-06-16", interval = "15m")

print(data.head)
