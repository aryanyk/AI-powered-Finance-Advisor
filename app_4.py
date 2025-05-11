import streamlit as st
import yfinance as yf
from prophet import Prophet
import matplotlib.pyplot as plt
import feedparser
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from scipy.special import softmax
import pandas as pd
import plotly.express as px

from google.generativeai import GenerativeModel
import google.generativeai as genai
import os

# Load FinBERT model once at startup
MODEL_NAME = "ProsusAI/finbert"
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME)

# Load Gemini API
os.environ["GOOGLE_API_KEY"] = "GEMINI_API_KEY"
genai.configure(api_key=os.environ["GOOGLE_API_KEY"])
gemini_model = GenerativeModel("gemini-2.0-flash")

# Streamlit app configuration
st.set_page_config(page_title="Stock Analysis Suite", layout="wide")
st.title("📈 Stock Prediction & News Sentiment Analysis")

# Sidebar inputs
st.sidebar.header("User Input")
ticker = st.sidebar.text_input("Enter stock ticker symbol (e.g., AAPL):", "AAPL")
years = st.sidebar.slider("Years of historical data:", 1, 10, 5)
forecast_years = st.sidebar.slider("Years to forecast:", 1, 5, 1)

def get_company_data(ticker):
    # Fetch the stock data
    stock = yf.Ticker(ticker)
    
    # Get basic information about the company
    info = stock.info

    # Display company details in the app
    st.subheader(f"📊 Company Overview for {ticker}")
    st.write(f"**Company Name**: {info.get('longName', 'N/A')}")
    st.write(f"**Sector**: {info.get('sector', 'N/A')}")
    st.write(f"**Industry**: {info.get('industry', 'N/A')}")
    st.write(f"**Country**: {info.get('country', 'N/A')}")
    st.write(f"**Market Cap**: {info.get('marketCap', 'N/A')}")
    st.write(f"**PE Ratio**: {info.get('trailingPE', 'N/A')}")
    st.write(f"**Dividend Yield**: {info.get('dividendYield', 'N/A')}")
    st.write(f"**52 Week High**: {info.get('fiftyTwoWeekHigh', 'N/A')}")
    st.write(f"**52 Week Low**: {info.get('fiftyTwoWeekLow', 'N/A')}")
    st.write(f"**Previous Close**: {info.get('regularMarketPreviousClose', 'N/A')}")
    st.write(f"**Current Price**: {info.get('regularMarketPrice', 'N/A')}")
    st.write(f"**Beta**: {info.get('beta', 'N/A')}")
    
    return stock

def main():
    if ticker:
        # Fetch and display company data
        stock = get_company_data(ticker)

        # Show Today's Stock Price (Intraday)
        st.header(f"💹 Today's Stock Price for {ticker}")
        try:
            intraday = yf.download(tickers=ticker, period='1d', interval='1m', progress=False)
            if not intraday.empty and 'Close' in intraday.columns:
                close_prices = intraday['Close'].dropna()
                if not close_prices.empty:
                    latest_price = float(close_prices.iloc[-1])
                    open_price = float(close_prices.iloc[0])
                    price_change = latest_price - open_price
                    percent_change = (price_change / open_price) * 100
                    latest_time = close_prices.index[-1].strftime('%Y-%m-%d %H:%M')

                    st.metric(
                        label=f"Last Updated: {latest_time}",
                        value=f"${latest_price:,.2f}",
                        delta=f"{price_change:+.2f} ({percent_change:+.2f}%)"
                    )
                    st.line_chart(close_prices, use_container_width=True)
                else:
                    st.warning("No valid price data available.")
            else:
                st.warning("Intraday data not available.")
        except Exception as e:
            st.error(f"Error fetching today's price: {e}")

        # Stock Prediction Section
        st.header(f"{ticker} Stock Price Prediction")

        with st.spinner("Analyzing historical data and generating forecast..."):
            try:
                data = yf.download(ticker, period=f"{years}y")
                df = data.reset_index()[['Date', 'Close']]
                df.columns = ['ds', 'y']

                m = Prophet(
                    yearly_seasonality=True,
                    weekly_seasonality=True,
                    daily_seasonality=False,
                    changepoint_prior_scale=0.05
                )
                m.add_country_holidays(country_name='US')
                m.fit(df)

                future = m.make_future_dataframe(periods=365 * forecast_years)
                forecast = m.predict(future)

                fig1 = m.plot(forecast)
                st.pyplot(fig1)

                st.subheader("Forecast Breakdown")
                fig2 = m.plot_components(forecast)
                st.pyplot(fig2)

            except Exception as e:
                st.error(f"Error in stock prediction: {str(e)}")

        # News Analysis Section
        st.header(f"📰 Latest News Sentiment Analysis for {ticker}")

        all_sentiments = []
        with st.spinner("Scanning financial news and analyzing sentiment..."):
            try:
                news_list = fetch_news(ticker)

                if news_list:
                    cols = st.columns(3)
                    for idx, news in enumerate(news_list):
                        sentiment, confidence = analyze_sentiment(news["title"])
                        all_sentiments.append(sentiment.split()[0])

                        with cols[idx % 3]:
                            st.markdown(f"### {news['source']}")
                            st.markdown(f"**{news['title']}**")
                            st.caption(f"Published: {news['published']}")

                            if "Bullish" in sentiment:
                                st.success(f"{sentiment} ({max(confidence):.0%})")
                            elif "Bearish" in sentiment:
                                st.error(f"{sentiment} ({max(confidence):.0%})")
                            else:
                                st.info(f"{sentiment} ({max(confidence):.0%})")

                            st.markdown(f"[Read Article]({news['url']})")
                            st.markdown("---")
                else:
                    st.warning("No recent news articles found for this ticker")
            except Exception as e:
                st.error(f"Error in news analysis: {str(e)}")

        # Sentiment Pie Chart
        st.subheader("📊 News Sentiment Distribution")
        if all_sentiments:
            sentiment_df = pd.DataFrame(all_sentiments, columns=['Sentiment'])
            pie_fig = px.pie(sentiment_df, names='Sentiment', title='Sentiment Distribution')
            st.plotly_chart(pie_fig)

        # Historical Stock Statistics
        st.subheader("📉 Historical Stock Statistics")
        if not data.empty:
            returns = data['Close'].pct_change().dropna()
            risk_score = float(returns.std() * 100)
            avg_return = float(returns.mean() * 100)

            st.metric("Risk Score (Volatility %)", f"{risk_score:.2f}%")
            st.metric("Average Daily Return (%)", f"{avg_return:.2f}%")

            hist_fig = px.histogram(returns, nbins=50, title="Daily Return Distribution")
            st.plotly_chart(hist_fig)

        # Final Recommendation (Revised)
        st.subheader("🧠 AI Investment Recommendation")
        try:
            prompt = f"""
            You are a financial advisor. Based on the stock {ticker}, the current market conditions are as follows:
            - News Sentiment Distribution: {all_sentiments}
            - Risk Score: {risk_score:.2f}%
            - Average Daily Return: {avg_return:.2f}%

            Please analyze:
            1. Key strengths of the company.
            2. Key weaknesses or risks.
            3. A final investment recommendation (Buy, Hold, or Sell) with reasoning.

            Keep the response structured in markdown with clear sections.
            """

            gemini_response = gemini_model.generate_content(prompt)
            response_text = gemini_response.text

            if response_text:
                st.markdown("### 📌 Summary Recommendation")
                st.markdown(response_text)

                # Optional: Extract a simplified recommendation for UI emphasis
                if "Buy" in response_text:
                    st.success("✅ Investment Decision: BUY")
                elif "Hold" in response_text:
                    st.info("⏸ Investment Decision: HOLD")
                elif "Sell" in response_text:
                    st.error("⚠️ Investment Decision: SELL")
                else:
                    st.warning("No clear decision found.")
            else:
                st.warning("Gemini did not return a response.")
        except Exception as e:
            st.error(f"Error generating recommendation: {e}")

def fetch_news(ticker):
    news_items = []

    rss_feeds = {
        "Yahoo Finance": f"https://feeds.finance.yahoo.com/rss/2.0/headline?s={ticker}&region=US&lang=en-US",
        "Google Finance": f"https://news.google.com/rss/search?q={ticker}+stock",
        "Reuters": f"https://www.reuters.com/companies/{ticker}/news/rss"
    }

    for source, url in rss_feeds.items():
        feed = feedparser.parse(url)
        for entry in feed.entries[:3]:
            news_items.append({
                "source": source,
                "title": entry.title,
                "published": entry.get('published', 'N/A'),
                "url": entry.link
            })

    return news_items

def analyze_sentiment(text):
    inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=512)
    with torch.no_grad():
        outputs = model(**inputs).logits
    probs = softmax(outputs.numpy().flatten())
    labels = ["Neutral ⚪", "Bullish 🟢", "Bearish 🔴"]
    return labels[probs.argmax()], probs

if __name__ == "__main__":
    main()
