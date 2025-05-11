# AI-Powered Financial Advisor  

## Overview  
This project is an **AI-powered financial advisor** designed to assist users in making informed stock market decisions. It provides recommendations on whether to **hold**, **sell**, or **buy** stocks by leveraging advanced machine learning models, sentiment analysis, and financial data integration.  

## Key Features  
- **LSTM Model**: Utilizes Long Short-Term Memory (LSTM) networks for time-series forecasting to predict stock price trends.  
- **Prophet Library**: Integrates Meta's Prophet library for robust trend analysis and seasonality detection.  
- **FINBERT Sentiment Analysis**: Employs FINBERT, a financial sentiment analysis model, to analyze market news and sentiment around specific stocks.  
- **SWOT Analysis**: Generates a comprehensive SWOT (Strengths, Weaknesses, Opportunities, Threats) analysis for stocks using data from the GEMINI API.  
- **Decision Support**: Combines all insights to provide actionable recommendations for stock trading decisions.  

## How It Works  
1. **Data Collection**:  
    - Fetches historical stock data and real-time updates using the GEMINI API.  
    - Gathers financial news and sentiment data for analysis.  

2. **Analysis Pipeline**:  
    - **LSTM Model** predicts future stock prices based on historical trends.  
    - **Prophet Library** identifies seasonal patterns and long-term trends.  
    - **FINBERT** analyzes sentiment from financial news to gauge market mood.  

3. **SWOT Analysis**:  
    - Integrates data to evaluate the stock's strengths, weaknesses, opportunities, and threats.  

4. **Decision Making**:  
    - Combines all insights to recommend whether to hold, sell, or buy a stock.  

## Technologies Used  
- **Python**  
- **TensorFlow/Keras** for LSTM implementation  
- **Prophet** by Meta for trend analysis  
- **FINBERT** for sentiment analysis  
- **GEMINI API** for financial data integration  

## Benefits  
- Provides data-driven insights for stock trading decisions.  
- Reduces emotional bias in trading.  
- Combines technical analysis with sentiment analysis for a holistic view.  

## Getting Started  
1. Clone the repository:  
    ```bash  
    git clone https://github.com/your-repo/ai-financial-advisor.git  
    ```  
2. Install dependencies:  
    ```bash  
    pip install -r requirements.txt  
    ```  
3. Run the application:  
    ```bash  
    python app.py  
    ```  

## Future Enhancements  
- Integration with additional APIs for broader data coverage.  
- Support for cryptocurrency analysis.  
- Enhanced visualization of SWOT analysis and predictions.  

## License  
This project is licensed under the [MIT License](LICENSE).  

## Contributing  
Contributions are welcome! Please open an issue or submit a pull request for any improvements or bug fixes.  

---  
**Disclaimer**: This tool is for informational purposes only and does not constitute financial advice. Always consult a professional financial advisor before making investment decisions.  