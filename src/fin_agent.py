import json
import numpy as np
import pandas as pd
from scipy.stats import spearmanr

from agno.tools import Toolkit
import yfinance as yf

class YFinanceTools(Toolkit):
    def __init__(
        self,
        stock_price: bool = True,
        company_info: bool = False,
        stock_fundamentals: bool = False,
        income_statements: bool = False,
        key_financial_ratios: bool = False,
        analyst_recommendations: bool = False,
        company_news: bool = False,
        technical_indicators: bool = False,
        historical_prices: bool = False,
        correlation: bool = False,
        volatility: bool = True,
        enable_all: bool = False,
    ):
        super().__init__(name="yfinance_tools")

        if stock_price or enable_all:
            self.register(self.get_current_stock_price)
        if company_info or enable_all:
            self.register(self.get_company_info)
        if stock_fundamentals or enable_all:
            self.register(self.get_stock_fundamentals)
        if income_statements or enable_all:
            self.register(self.get_income_statements)
        if key_financial_ratios or enable_all:
            self.register(self.get_key_financial_ratios)
        if analyst_recommendations or enable_all:
            self.register(self.get_analyst_recommendations)
        if company_news or enable_all:
            self.register(self.get_company_news)
        if technical_indicators or enable_all:
            self.register(self.get_technical_indicators)
        if historical_prices or enable_all:
            self.register(self.get_historical_stock_prices)
        if correlation or enable_all:
            self.register(self.get_correlation)
        if volatility or enable_all:
            self.register(self.get_volatility)

    def get_current_stock_price(self, symbol: str) -> str:
        """
        Use this function to get the current stock price for a given symbol.

        Args:
            symbol (str): The stock symbol.

        Returns:
            str: The current stock price or error message.
        """
        try:
            stock = yf.Ticker(symbol)
            # Use "regularMarketPrice" for regular market hours, or "currentPrice" for pre/post market
            current_price = stock.info.get("regularMarketPrice", stock.info.get("currentPrice"))
            return f"{current_price:.4f}" if current_price else f"Could not fetch current price for {symbol}"
        except Exception as e:
            return f"Error fetching current price for {symbol}: {e}"

    def get_company_info(self, symbol: str) -> str:
        """Use this function to get company information and overview for a given stock symbol.

        Args:
            symbol (str): The stock symbol.

        Returns:
            str: JSON containing company profile and overview.
        """
        try:
            company_info_full = yf.Ticker(symbol).info
            if company_info_full is None:
                return f"Could not fetch company info for {symbol}"

            company_info_cleaned = {
                "Name": company_info_full.get("shortName"),
                "Symbol": company_info_full.get("symbol"),
                "Current Stock Price": f"{company_info_full.get('regularMarketPrice', company_info_full.get('currentPrice'))} {company_info_full.get('currency', 'USD')}",
                "Market Cap": f"{company_info_full.get('marketCap', company_info_full.get('enterpriseValue'))} {company_info_full.get('currency', 'USD')}",
                "Sector": company_info_full.get("sector"),
                "Industry": company_info_full.get("industry"),
                "Address": company_info_full.get("address1"),
                "City": company_info_full.get("city"),
                "State": company_info_full.get("state"),
                "Zip": company_info_full.get("zip"),
                "Country": company_info_full.get("country"),
                "EPS": company_info_full.get("trailingEps"),
                "P/E Ratio": company_info_full.get("trailingPE"),
                "52 Week Low": company_info_full.get("fiftyTwoWeekLow"),
                "52 Week High": company_info_full.get("fiftyTwoWeekHigh"),
                "50 Day Average": company_info_full.get("fiftyDayAverage"),
                "200 Day Average": company_info_full.get("twoHundredDayAverage"),
                "Website": company_info_full.get("website"),
                "Summary": company_info_full.get("longBusinessSummary"),
                "Analyst Recommendation": company_info_full.get("recommendationKey"),
                "Number Of Analyst Opinions": company_info_full.get("numberOfAnalystOpinions"),
                "Employees": company_info_full.get("fullTimeEmployees"),
                "Total Cash": company_info_full.get("totalCash"),
                "Free Cash flow": company_info_full.get("freeCashflow"),
                "Operating Cash flow": company_info_full.get("operatingCashflow"),
                "EBITDA": company_info_full.get("ebitda"),
                "Revenue Growth": company_info_full.get("revenueGrowth"),
                "Gross Margins": company_info_full.get("grossMargins"),
                "Ebitda Margins": company_info_full.get("ebitdaMargins"),
            }
            return json.dumps(company_info_cleaned, indent=2)
        except Exception as e:
            return f"Error fetching company profile for {symbol}: {e}"

    def get_historical_stock_prices(self, symbol: str, period: str = "1mo", interval: str = "1d") -> str:
        """
        Use this function to get the historical stock price for a given symbol.

        Args:
            symbol (str): The stock symbol.
            period (str): The period for which to retrieve historical prices. Defaults to "1mo".
                        Valid periods: 1d,5d,1mo,3mo,6mo,1y,2y,5y,10y,ytd,max
            interval (str): The interval between data points. Defaults to "1d".
                        Valid intervals: 1d,5d,1wk,1mo,3mo

        Returns:
          str: The current stock price or error message.
        """
        try:
            stock = yf.Ticker(symbol)
            historical_price = stock.history(period=period, interval=interval)
            return historical_price.to_json(orient="index")
        except Exception as e:
            return f"Error fetching historical prices for {symbol}: {e}"

    def get_stock_fundamentals(self, symbol: str) -> str:
        """Use this function to get fundamental data for a given stock symbol yfinance API.

        Args:
            symbol (str): The stock symbol.

        Returns:
            str: A JSON string containing fundamental data or an error message.
                Keys:
                    - 'symbol': The stock symbol.
                    - 'company_name': The long name of the company.
                    - 'sector': The sector to which the company belongs.
                    - 'industry': The industry to which the company belongs.
                    - 'market_cap': The market capitalization of the company.
                    - 'pe_ratio': The forward price-to-earnings ratio.
                    - 'pb_ratio': The price-to-book ratio.
                    - 'dividend_yield': The dividend yield.
                    - 'eps': The trailing earnings per share.
                    - 'beta': The beta value of the stock.
                    - '52_week_high': The 52-week high price of the stock.
                    - '52_week_low': The 52-week low price of the stock.
        """
        try:
            stock = yf.Ticker(symbol)
            info = stock.info
            fundamentals = {
                "symbol": symbol,
                "company_name": info.get("longName", ""),
                "sector": info.get("sector", ""),
                "industry": info.get("industry", ""),
                "market_cap": info.get("marketCap", "N/A"),
                "pe_ratio": info.get("forwardPE", "N/A"),
                "pb_ratio": info.get("priceToBook", "N/A"),
                "dividend_yield": info.get("dividendYield", "N/A"),
                "eps": info.get("trailingEps", "N/A"),
                "beta": info.get("beta", "N/A"),
                "52_week_high": info.get("fiftyTwoWeekHigh", "N/A"),
                "52_week_low": info.get("fiftyTwoWeekLow", "N/A"),
            }
            return json.dumps(fundamentals, indent=2)
        except Exception as e:
            return f"Error getting fundamentals for {symbol}: {e}"

    def get_income_statements(self, symbol: str) -> str:
        """Use this function to get income statements for a given stock symbol.

        Args:
            symbol (str): The stock symbol.

        Returns:
            dict: JSON containing income statements or an empty dictionary.
        """
        try:
            stock = yf.Ticker(symbol)
            financials = stock.financials
            return financials.to_json(orient="index")
        except Exception as e:
            return f"Error fetching income statements for {symbol}: {e}"

    def get_key_financial_ratios(self, symbol: str) -> str:
        """Use this function to get key financial ratios for a given stock symbol.

        Args:
            symbol (str): The stock symbol.

        Returns:
            dict: JSON containing key financial ratios.
        """
        try:
            stock = yf.Ticker(symbol)
            key_ratios = stock.info
            return json.dumps(key_ratios, indent=2)
        except Exception as e:
            return f"Error fetching key financial ratios for {symbol}: {e}"

    def get_analyst_recommendations(self, symbol: str) -> str:
        """Use this function to get analyst recommendations for a given stock symbol.

        Args:
            symbol (str): The stock symbol.

        Returns:
            str: JSON containing analyst recommendations.
        """
        try:
            stock = yf.Ticker(symbol)
            recommendations = stock.recommendations
            return recommendations.to_json(orient="index")
        except Exception as e:
            return f"Error fetching analyst recommendations for {symbol}: {e}"

    def get_company_news(self, symbol: str, num_stories: int = 3) -> str:
        """Use this function to get company news and press releases for a given stock symbol.

        Args:
            symbol (str): The stock symbol.
            num_stories (int): The number of news stories to return. Defaults to 3.

        Returns:
            str: JSON containing company news and press releases.
        """
        try:
            news = yf.Ticker(symbol).news
            return json.dumps(news[:num_stories], indent=2)
        except Exception as e:
            return f"Error fetching company news for {symbol}: {e}"

    # def get_correlation(self, symbols: list, period: str = "1y", interval: str = "1d") -> str:
    #     """Calculate Pearson and Spearman correlations between multiple assets efficiently."""
    #     if not isinstance(symbols, list) or len(symbols) < 2:
    #         return json.dumps({"error": "At least two symbols are required for correlation analysis."}, indent=2)

    #     try:
    #         tickers = yf.Tickers(symbols)
    #         df = tickers.history(period=period, interval=interval, actions=False)['Close']
    #     except ValueError as e:
    #         return json.dumps({"error": f"Invalid symbol(s) or fetch error: {str(e)}"}, indent=2)
    #     except Exception as e:
    #         return json.dumps({"error": f"Invalid symbol(s) or fetch error: {str(e)}"}, indent=2)

    #     returns = df.pct_change().dropna(how='any')
    #     if returns.empty or len(returns) < 2:
    #         return json.dumps({"error": f"Invalid symbol(s) or fetch error: {str(e)}"}, indent=2)
        
    #     pearson_corr = returns.corr(method='pearson').replace(np.nan, 0.0).to_dict()

    #     ranks = returns.rank().values
    #     spearman_corr = pd.DataFrame(
    #         np.corrcoef(ranks.T),
    #         index=returns.columns,
    #         columns=returns.columns
    #     ).replace(np.nan, 0.0).to_dict()

    #     # Compile metadata
    #     metadata = {
    #         "symbols": list(returns.columns),
    #         "period": period,
    #         "interval": interval,
    #         "start_date": returns.index[0].strftime('%Y-%m-%d'),
    #         "end_date": returns.index[-1].strftime('%Y-%m-%d'),
    #         "data_points": len(returns),
    #         "pearson_correlations": pearson_corr,
    #         "spearman_correlations": spearman_corr
    #     }

    #     return json.dumps(metadata, indent=2)
    
    def get_correlation(self, symbol_1: str, symbol_2: str, period: str = "1y", interval: str = "1d") -> str:
        """Use this function to get company correlation between two assets

        Args:
            symbol_1 (str): The stock symbol.
            symbol_2 (str): The stock symbol.

        Returns:
            str: JSON containing company news and press releases.
        """        
        try:
            data1 = yf.Ticker(symbol_1).history(period=period, interval=interval)['Close']
            data2 = yf.Ticker(symbol_2).history(period=period, interval=interval)['Close']
            
            returns1 = data1.pct_change().dropna()
            returns2 = data2.pct_change().dropna()
            
            aligned_data = returns1.align(returns2, join='inner')
            returns1_aligned, returns2_aligned = aligned_data

            pearson_corr = returns1_aligned.corr(returns2_aligned)
            
            spearman_corr, _ = spearmanr(returns1_aligned, returns2_aligned)
            
            metadata = {
                "symbol_1": symbol_1,
                "symbol_2": symbol_2,
                "period": period,
                "interval": interval,
                "start_date": str(data1.index[0]),
                "end_date": str(data1.index[-1]),
                "data_points": len(returns1_aligned),
                "pearson_corr": pearson_corr,
                "spearman_corr": spearman_corr
            }
            return json.dumps(metadata, indent=2)

        except Exception as e:
            return f"Error fetching company news for {symbol_1} - {symbol_2}: {e}"
        

    def get_volatility(self, symbol: str, period: str = "1y", interval: str = "1d", benchmark_symbol: str = "^SPX") -> str:
        """
        Fetch and calculate enhanced volatility metrics for a given stock symbol, including Sharpe Ratio, Max Drawdown, and Beta.

        Args:
            symbol (str): The stock symbol (e.g., 'AAPL').
            period (str): The time period for data retrieval.
                Valid periods: 1d, 5d, 1mo, 3mo, 6mo, 1y, 2y, 5y, 10y, ytd, max. Defaults to 1y.
            interval (str): The data interval (e.g., '1d' for daily). Defaults to '1d'.
            benchmark_symbol (str): Benchmark for Beta calculation (e.g., 'SPY'). Defaults to 'SPY'.

        Returns:
            str: JSON string containing volatility metrics or an error message.
        """

        try:
            ticker = yf.Ticker(symbol)
            benchmark = yf.Ticker(benchmark_symbol)
            data = ticker.history(period=period, interval=interval)
            benchmark_data = benchmark.history(period=period, interval=interval)

            if data.empty or benchmark_data.empty:
                return json.dumps({"error": f"No data available for {symbol} or {benchmark_symbol} with period={period}, interval={interval}"}, indent=2)

            info = ticker.info
            current_price = info.get("regularMarketPrice", data["Close"].iloc[-1])
            target_price = info.get("targetMeanPrice", current_price)

            returns = data['Close'].pct_change().dropna()
            benchmark_returns = benchmark_data['Close'].pct_change().dropna()

            aligned_returns = returns.align(benchmark_returns, join='inner')
            asset_returns, benchmark_returns = aligned_returns

            volatility_std = returns.std() * (252 ** 0.5)

            high_low = data['High'] - data['Low']
            high_close = (data['High'] - data['Close'].shift()).abs()
            low_close = (data['Low'] - data['Close'].shift()).abs()
            true_range = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
            atr = true_range.rolling(window=14, min_periods=1).mean().iloc[-1]

            annualized_return = returns.mean() * 252  # Annualized mean return
            sharpe_ratio = annualized_return / volatility_std if volatility_std != 0 else 0

            # Maximum Drawdown
            cumulative_returns = (1 + returns).cumprod()
            peak = cumulative_returns.cummax()
            drawdown = (cumulative_returns - peak) / peak
            max_drawdown = drawdown.min()

            covariance = asset_returns.cov(benchmark_returns)
            benchmark_variance = benchmark_returns.var()
            beta = covariance / benchmark_variance if benchmark_variance != 0 else 0


            upside_potential = ((target_price - current_price) / current_price) * 100 if current_price != 0 else 0
            risk_reward_ratio = abs(upside_potential / (max_drawdown * 100)) if max_drawdown != 0 else 0

            # Metadata and results
            metadata = {
                "symbol": symbol,
                "benchmark_symbol": benchmark_symbol,
                "period": period,
                "interval": interval,
                "start_date": str(data.index[0].date()),
                "end_date": str(data.index[-1].date()),
                "data_points": len(returns),
                "volatility_std_annualized": f"{(volatility_std * 100).round(2)}%",
                "atr_latest": atr.round(4),
                "sharpe_ratio": sharpe_ratio.round(4),
                "max_drawdown": f"{(max_drawdown * 100).round(2)}%",
                "beta": round(beta, 2),
                "risk_reward_ratio": round(risk_reward_ratio, 2),
                "current_price": round(current_price, 2),
                "target_price": round(target_price, 2),
                "upside_potential_percent": round(upside_potential, 2),
            }
            return json.dumps(metadata, indent=2)

        except Exception as e:
            return json.dumps({"error": f"Error fetching volatility for {symbol}: {str(e)}"}, indent=2)

    def get_technical_indicators(self, symbol: str, period: str = "1y", interval: str = "1d") -> str:
        """
        Get a comprehensive set of technical indicators for a given stock symbol.

        Args:
            symbol (str): The stock symbol (e.g., 'SPY').
            period (str): The time period for data retrieval.
                Valid periods: 1d, 5d, 1mo, 3mo, 6mo, 1y, 2y, 5y, 10y, ytd, max. Defaults to 3mo.
            interval (str): The data interval (e.g., '1d' for daily). Defaults to '1d'.

        Returns:
            str: JSON string containing technical indicators or an error message.
        """
        try:
            # Fetch historical data
            ticker = yf.Ticker(symbol)
            data = ticker.history(period=period, interval=interval)

            if data.empty:
                return json.dumps({"error": f"No data available for {symbol} with period={period}, interval={interval}"}, indent=2)

            # Standardize column names
            data = data.rename(columns={"Open": "open", "High": "high", "Low": "low", "Close": "close", "Volume": "volume"})

            # Moving Averages
            data["sma_20"] = data["close"].rolling(window=20).mean()
            data["sma_50"] = data["close"].rolling(window=50).mean()
            data["ema_12"] = data["close"].ewm(span=12, adjust=False).mean()

            # RSI
            delta = data["close"].diff()
            gain = delta.where(delta > 0, 0).rolling(window=14).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
            rs = gain / loss.replace(0, pd.NA)  # Avoid division by zero
            data["rsi"] = 100 - (100 / (1 + rs))

            # MACD
            ema_12 = data["close"].ewm(span=12, adjust=False).mean()
            ema_26 = data["close"].ewm(span=26, adjust=False).mean()
            data["macd"] = ema_12 - ema_26
            data["signal_line"] = data["macd"].ewm(span=9, adjust=False).mean()
            data["macd_histogram"] = data["macd"] - data["signal_line"]

            # Bollinger Bands
            data["bb_middle"] = data["close"].rolling(window=20).mean()
            bb_std = data["close"].rolling(window=20).std()
            data["bb_upper"] = data["bb_middle"] + 2 * bb_std
            data["bb_lower"] = data["bb_middle"] - 2 * bb_std

            # VWAP (cumulative over period)
            typical_price = (data["high"] + data["low"] + data["close"]) / 3
            data["vwap"] = (typical_price * data["volume"]).cumsum() / data["volume"].cumsum()

            # ATR (Average True Range)
            tr = pd.concat([
                data["high"] - data["low"],
                (data["high"] - data["close"].shift()).abs(),
                (data["low"] - data["close"].shift()).abs()
            ], axis=1).max(axis=1)
            data["atr"] = tr.rolling(window=14).mean()

            # Stochastic Oscillator
            low_14 = data["low"].rolling(window=14).min()
            high_14 = data["high"].rolling(window=14).max()
            data["stoch_k"] = 100 * (data["close"] - low_14) / (high_14 - low_14)
            data["stoch_d"] = data["stoch_k"].rolling(window=3).mean()

            # ADX (Average Directional Index)
            plus_dm = (data["high"] - data["high"].shift()).where(lambda x: x > 0, 0)
            minus_dm = (data["low"].shift() - data["low"]).where(lambda x: x > 0, 0)
            tr_smooth = tr.rolling(window=14).mean()
            plus_di = 100 * (plus_dm.rolling(window=14).mean() / tr_smooth)
            minus_di = 100 * (minus_dm.rolling(window=14).mean() / tr_smooth)
            dx = 100 * abs(plus_di - minus_di) / (plus_di + minus_di)
            data["adx"] = dx.rolling(window=14).mean()

            # On-Balance Volume (OBV)
            data["obv"] = (data["volume"] * ((data["close"] > data["close"].shift()).astype(int) - 
                                            (data["close"] < data["close"].shift()).astype(int))).cumsum()

            # Select indicators
            indicators = data[[
                "close", "sma_20", "sma_50", "ema_12", "rsi", 
                "macd", "signal_line", "macd_histogram", 
                "bb_upper", "bb_middle", "bb_lower", "vwap", "atr",
                "stoch_k", "stoch_d", "adx", "obv", "volume"
            ]]

            # Format index and drop NaN
            indicators.index = indicators.index.strftime("%Y-%m-%d")
            result = indicators.dropna().to_dict(orient="index")

            # Metadata
            metadata = {
                "symbol": symbol,
                "period": period,
                "interval": interval,
                "start_date": indicators.index[0],
                "end_date": indicators.index[-1],
                "data_points": len(indicators)
            }

            return json.dumps({"metadata": metadata, "indicators": result}, indent=2)

        except Exception as e:
            return json.dumps({"error": f"Error fetching technical indicators for {symbol}: {str(e)}"}, indent=2)

    def get_historical_comparison(self, symbol: str, years: int = 5) -> str:
        """
        Compare a stock's current P/E ratio, dividend yield, and revenue growth to their historical averages.

        Args:
            symbol (str): The stock ticker symbol (e.g., 'AAPL' for Apple).
            years (int): Number of years for historical data (default is 5).

        Returns:
            str: A JSON string with current metrics, historical averages, and comparisons.
                 Returns an error message in JSON format if data retrieval fails.
        """
        try:
            stock = yf.Ticker(symbol)

            financials = stock.financials
            if financials.empty:
                return json.dumps({"error": f"No financial data available for {symbol}"}, indent=2)
            
            financials = financials.loc[:, financials.notna().sum() > 5].copy()

            fiscal_years = financials.columns
            dates = [pd.to_datetime(date) for date in fiscal_years]

            if len(dates) < 2:
                return json.dumps({"error": f"Insufficient historical data for {symbol}"}, indent=2)

            eps_history = financials.loc['Basic EPS'] if 'Basic EPS' in financials.index else None
            revenue_history = financials.loc['Total Revenue'] if 'Total Revenue' in financials.index else None
            if eps_history is None or revenue_history is None:
                return json.dumps({"error": f"Missing EPS or revenue data for {symbol}"}, indent=2)

            prices = {}
            for date in dates:
                hist = stock.history(start=date - pd.Timedelta(days=30), end=date + pd.Timedelta(days=1))
                if not hist.empty:
                    prices[date] = hist['Close'].iloc[-1]
                else:
                    prices[date] = None

            pe_history = {}
            for date in dates:
                eps = eps_history.get(date)
                price = prices.get(date)
                if price is not None and eps is not None and eps != 0:
                    pe_history[date] = price / eps
                else:
                    pe_history[date] = None

            valid_pe = [pe for pe in pe_history.values() if pe is not None]
            avg_pe = sum(valid_pe) / len(valid_pe) if valid_pe else None

            dividends = stock.dividends
            dividends_by_year = dividends.groupby(dividends.index.year).sum()
            div_yield_history = {}
            for year in range(max(dates[0].year, dividends.index.year.min() if not dividends.empty else dates[0].year), dates[-1].year + 1):
                div_year = dividends_by_year.get(year, 0)
                dec31 = pd.to_datetime(f'{year}-12-31').date()
                hist = stock.history(start=dec31 - pd.Timedelta(days=30), end=dec31 + pd.Timedelta(days=1))
                if not hist.empty:
                    price_dec31 = hist['Close'].iloc[-1]
                    div_yield_history[year] = (div_year / price_dec31) * 100 if price_dec31 != 0 else None
                else:
                    div_yield_history[year] = None

            valid_div_yield = [dy for dy in div_yield_history.values() if dy is not None]
            avg_div_yield = sum(valid_div_yield) / len(valid_div_yield) if valid_div_yield else None

            revenue_growth_history = {}
            for i in range(1, len(revenue_history)):
                prev_rev = revenue_history.iloc[i - 1]
                curr_rev = revenue_history.iloc[i]
                if prev_rev != 0:
                    revenue_growth_history[fiscal_years[i]] = ((curr_rev - prev_rev) / prev_rev) * 100
                else:
                    revenue_growth_history[fiscal_years[i]] = None

            valid_growth = [g for g in revenue_growth_history.values() if g is not None]
            avg_revenue_growth = sum(valid_growth) / len(valid_growth) if valid_growth else None

            current_pe = stock.info.get('trailingPE', None)
            current_div_yield = stock.info.get('dividendYield', None)
            if current_div_yield is not None:
                current_div_yield *= 100
            current_revenue_growth = stock.info.get('revenueGrowth', None)
            if current_revenue_growth is not None:
                current_revenue_growth *= 100

            comparison = {}
            if current_pe is not None and avg_pe is not None:
                comparison['pe_ratio'] = {
                    'current': round(current_pe, 2),
                    'historical_average': round(avg_pe, 2),
                    'comparison': ('below average' if current_pe < avg_pe else
                                  'above average' if current_pe > avg_pe else 'average')
                }
            if current_div_yield is not None and avg_div_yield is not None:
                comparison['dividend_yield'] = {
                    'current': round(current_div_yield, 2),
                    'historical_average': round(avg_div_yield, 2),
                    'comparison': ('above average' if current_div_yield > avg_div_yield else
                                  'below average' if current_div_yield < avg_div_yield else 'average')
                }
            if current_revenue_growth is not None and avg_revenue_growth is not None:
                comparison['revenue_growth'] = {
                    'current': round(current_revenue_growth, 2),
                    'historical_average': round(avg_revenue_growth, 2),
                    'comparison': ('above average' if current_revenue_growth > avg_revenue_growth else
                                  'below average' if current_revenue_growth < avg_revenue_growth else 'average')
                }

            return json.dumps(comparison, indent=2)

        except Exception as e:
            return json.dumps({"error": f"Error processing {symbol}: {str(e)}"}, indent=2)