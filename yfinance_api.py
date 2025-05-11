import yfinance as yf

def get_company_data(ticker):
    # Fetch the stock data
    stock = yf.Ticker(ticker)
    
    # Get basic information about the company
    info = stock.info

    # Print the company details
    print(f"Company: {info.get('longName', 'N/A')}")
    print(f"Ticker: {ticker}")
    print(f"Sector: {info.get('sector', 'N/A')}")
    print(f"Industry: {info.get('industry', 'N/A')}")
    print(f"Country: {info.get('country', 'N/A')}")
    print(f"Market Cap: {info.get('marketCap', 'N/A')}")
    print(f"PE Ratio: {info.get('trailingPE', 'N/A')}")
    print(f"Dividend Yield: {info.get('dividendYield', 'N/A')}")
    print(f"52 Week High: {info.get('fiftyTwoWeekHigh', 'N/A')}")
    print(f"52 Week Low: {info.get('fiftyTwoWeekLow', 'N/A')}")
    print(f"Previous Close: {info.get('regularMarketPreviousClose', 'N/A')}")
    print(f"Current Price: {info.get('regularMarketPrice', 'N/A')}")
    print(f"Beta: {info.get('beta', 'N/A')}")
    
    # Fetch historical data (last 5 days)
    hist = stock.history(period="5d")
    print("\nHistorical Data (last 5 days):")
    print(hist[['Open', 'Close', 'High', 'Low', 'Volume']])

if __name__ == "__main__":
    ticker = input("Enter the stock ticker symbol (e.g., AAPL): ")
    get_company_data(ticker)
