import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import requests
import json
import os
from sklearn.preprocessing import MinMaxScaler
from sklearn.decomposition import PCA

# ─── GEMINI API CONFIGURATION ─────────────────────────────────────────────────
# Note: Hardcoding API keys is insecure for production. Use environment variables or Streamlit secrets instead.
GEMINI_API_KEY = "AIzaSyC_IhtlIwWU6WbHV0jd_L5z17gNGyuqFso"
GEMINI_API_URL = "https://generativelanguage.googleapis.com/v1beta/models/gemini-2.0-flash:generateContent"

def query_gemini_api(ticker, metric):
    """
    Query Gemini API for specific Management & Governance metric.
    Returns a numeric value or None if the query fails.
    """
    headers = {
        "Content-Type": "application/json",
    }
    prompt = f"Provide the latest {metric} for the company with ticker {ticker} listed on major stock exchanges. Return only a numeric value or a short description (e.g., 'Increasing', 'Strong') if applicable."
    data = {
        "contents": [{"parts": [{"text": prompt}]}]
    }
    try:
        response = requests.post(
            f"{GEMINI_API_URL}?key={GEMINI_API_KEY}",
            headers=headers,
            json=data
        )
        response.raise_for_status()
        result = response.json().get("candidates", [{}])[0].get("content", {}).get("parts", [{}])[0].get("text", "")
        if metric == "Promoter Holding Trend":
            if "Increasing" in result or "Stable" in result:
                return 5
            elif "Declining" in result and "<10%" in result:
                return 3
            elif "Sharp Decline" in result:
                return 1
            try:
                return float(result)
            except:
                return 3  # Neutral default
        elif metric == "Board & Audit Quality":
            if "Strong" in result:
                return 5
            elif "Moderate" in result:
                return 3
            elif "Weak" in result:
                return 1
            try:
                return float(result)
            except:
                return 3  # Neutral default
        else:  # Pledged Shares %
            try:
                return float(result.replace("%", ""))
            except:
                return 0.0  # Default to 0%
    except Exception as e:
        st.warning(f"Failed to fetch {metric} for {ticker} via Gemini API: {str(e)}")
        return None

# ─── HELPER FUNCTIONS ────────────────────────────────────────────────────────
def get_stock_data(ticker):
    try:
        stock = yf.Ticker(ticker)
        hist = stock.history(period="20y")
        if hist.empty:
            st.warning(f"No historical data found for {ticker}")
            return None
        try:
            financials = stock.financials
            balance_sheet = stock.balance_sheet
            cashflow = stock.cashflow
            institutional_holders = stock.institutional_holders
        except Exception as e:
            st.warning(f"Could not fetch financials or holders for {ticker}: {str(e)}")
            financials = pd.DataFrame()
            balance_sheet = pd.DataFrame()
            cashflow = pd.DataFrame()
            institutional_holders = pd.DataFrame()
        try:
            info = stock.info
        except:
            info = {}
        return {
            'stock': stock,
            'ticker': ticker,
            'hist': hist,
            'financials': financials,
            'balance_sheet': balance_sheet,
            'cashflow': cashflow,
            'info': info,
            'institutional_holders': institutional_holders
        }
    except Exception as e:
        st.error(f"Error fetching data for {ticker}: {str(e)}")
        return None

def calculate_fundamentals(data):
    if not data:
        return None
    try:
        stock = data['stock']
        ticker = data['ticker']
        financials = data['financials']
        balance_sheet = data['balance_sheet']
        cashflow = data['cashflow']
        info = data['info']
        institutional_holders = data['institutional_holders']
        
        # Helper function to get metric from dataframe
        def get_metric(series_name, df, default=0.0):
            try:
                matches = [idx for idx in df.index if series_name.lower() in idx.lower()]
                if matches:
                    s = df.loc[matches[0]].dropna()
                    return float(s.iloc[0]) if not s.empty else default
                return default
            except:
                return default
        
        # Helper function to calculate CAGR
        def calculate_cagr(start_value, end_value, years):
            if start_value is None or end_value is None or start_value <= 0 or end_value <= 0:
                return None
            return (end_value / start_value) ** (1/years) - 1
        
        # Initialize result dictionary
        result = {
            'Ticker': ticker,
            # Financial Metrics
            'Sales_CAGR_5Y': None,
            'PAT_CAGR_5Y': None,
            'Debt_to_Equity': None,
            'PE_Ratio': None,
            'ROE': None,
            'CFO_Latest': None,
            'FCF_Latest': None,
            'Revenue_Growth_YoY': None,
            'Net_Income_Growth_YoY': None,
            'CFO_5Y_Avg': None,
            'FCF_5Y_Avg': None,
            # Management & Governance
            'Promoter_Holding_Trend': None,
            'Pledged_Shares_Percent': None,
            'Board_Audit_Quality': None,
            'Institutional_Holding': None,
            # Earnings Quality / Growth
            'Revenue_CAGR_3Y': None,
            'Operating_Profit_CAGR': None,
            'PAT_CAGR_3Y': None,
            'EPS_Growth_YoY': None,
            'Sales_Growth_YoY': None,
            # Profitability
            'ROCE': None,
            'Net_Profit_Margin': None,
            # Valuation
            'EV_EBITDA': None,
            'Price_to_Book': None,
            # Balance Sheet Quality
            'Interest_Coverage_Ratio': None,
            'Current_Ratio': None,
            # Cash Flow Quality
            'OCF_Net_Profit_Ratio': None,
            'FCF_Trend_Score': None,
            'Cash_Flow_Stability': None,
            'OCF_Debt_Ratio': None
        }
        
        # Get basic financial metrics
        result['Sales_CAGR_5Y'] = get_metric('revenue', financials)
        result['PAT_CAGR_5Y'] = get_metric('net income', financials)
        result['Debt_to_Equity'] = float(info.get('debtToEquity', 0))
        result['PE_Ratio'] = float(info.get('trailingPE', 0))
        result['ROE'] = float(info.get('returnOnEquity', 0)) * 100
        result['CFO_Latest'] = get_metric('operating activities', cashflow)
        result['FCF_Latest'] = get_metric('free cash flow', cashflow)
        result['Revenue_Growth_YoY'] = get_metric('revenue', financials)  # Placeholder
        result['Net_Income_Growth_YoY'] = get_metric('net income', financials)  # Placeholder
        result['CFO_5Y_Avg'] = get_metric('operating activities', cashflow)  # Placeholder
        result['FCF_5Y_Avg'] = get_metric('free cash flow', cashflow)  # Placeholder
        
        # Management & Governance metrics
        result['Promoter_Holding_Trend'] = query_gemini_api(ticker, "Promoter Holding Trend") or st.session_state.get(f"promoter_{ticker}", 3)
        result['Pledged_Shares_Percent'] = query_gemini_api(ticker, "Pledged Shares %") or st.session_state.get(f"pledged_{ticker}", 0.0)
        result['Board_Audit_Quality'] = query_gemini_api(ticker, "Board & Audit Quality") or st.session_state.get(f"board_{ticker}", 3)
        
        # Institutional Holding
        try:
            if not institutional_holders.empty:
                total_institutional = institutional_holders['% Out'].sum() * 100
                result['Institutional_Holding'] = total_institutional
            else:
                result['Institutional_Holding'] = 0.0
        except:
            result['Institutional_Holding'] = 0.0
        
        # Earnings Quality / Growth metrics
        # Revenue CAGR (3Y)
        try:
            if len(financials.columns) >= 3:
                revenue_3y_ago = float(financials.iloc[0, -3])
                revenue_latest = float(financials.iloc[0, 0])
                result['Revenue_CAGR_3Y'] = calculate_cagr(revenue_3y_ago, revenue_latest, 3) * 100
        except:
            pass
        
        # Operating Profit CAGR
        try:
            if len(financials.columns) >= 3:
                operating_profit_3y_ago = float(financials.loc['Operating Income'].iloc[0, -3])
                operating_profit_latest = float(financials.loc['Operating Income'].iloc[0, 0])
                result['Operating_Profit_CAGR'] = calculate_cagr(operating_profit_3y_ago, operating_profit_latest, 3) * 100
        except:
            pass
        
        # PAT CAGR (3Y)
        try:
            if len(financials.columns) >= 3:
                pat_3y_ago = float(financials.loc['Net Income'].iloc[0, -3])
                pat_latest = float(financials.loc['Net Income'].iloc[0, 0])
                result['PAT_CAGR_3Y'] = calculate_cagr(pat_3y_ago, pat_latest, 3) * 100
        except:
            pass
        
        # EPS Growth YoY
        try:
            if len(financials.columns) >= 2:
                eps_prev = float(info.get('epsPreviousYearQuarter', 0))
                eps_current = float(info.get('epsForward', 0))
                if eps_prev > 0:
                    result['EPS_Growth_YoY'] = ((eps_current - eps_prev) / eps_prev) * 100
        except:
            pass
        
        # Sales Growth YoY
        try:
            if len(financials.columns) >= 2:
                revenue_prev = float(financials.loc['Total Revenue'].iloc[0, 1])
                revenue_current = float(financials.loc['Total Revenue'].iloc[0, 0])
                if revenue_prev > 0:
                    result['Sales_Growth_YoY'] = ((revenue_current - revenue_prev) / revenue_prev) * 100
        except:
            pass
        
        # Profitability metrics
        # ROCE
        try:
            ebit = get_metric('ebit', financials)
            total_assets = get_metric('total assets', balance_sheet)
            current_liabilities = get_metric('total current liabilities', balance_sheet)
            if ebit and total_assets and current_liabilities:
                capital_employed = total_assets - current_liabilities
                if capital_employed > 0:
                    result['ROCE'] = (ebit / capital_employed) * 100
        except:
            pass
        
        # Net Profit Margin
        try:
            net_income = get_metric('net income', financials)
            revenue = get_metric('revenue', financials)
            if revenue > 0:
                result['Net_Profit_Margin'] = (net_income / revenue) * 100
        except:
            pass
        
        # Valuation metrics
        # EV/EBITDA
        try:
            market_cap = float(info.get('marketCap', 0))
            total_debt = get_metric('total debt', balance_sheet)
            cash = get_metric('cash', balance_sheet)
            ebitda = get_metric('ebitda', financials)
            
            if market_cap and ebitda > 0:
                ev = market_cap + total_debt - cash
                result['EV_EBITDA'] = ev / ebitda
        except:
            pass
        
        # Price-to-Book
        try:
            book_value_per_share = float(info.get('bookValue', 0))
            current_price = float(info.get('currentPrice', 0))
            if book_value_per_share > 0:
                result['Price_to_Book'] = current_price / book_value_per_share
        except:
            pass
        
        # Balance Sheet Quality metrics
        # Interest Coverage Ratio
        try:
            ebit = get_metric('ebit', financials)
            interest_expense = get_metric('interest expense', financials)
            if interest_expense > 0:
                result['Interest_Coverage_Ratio'] = ebit / interest_expense
        except:
            pass
        
        # Current Ratio
        try:
            current_assets = get_metric('total current assets', balance_sheet)
            current_liabilities = get_metric('total current liabilities', balance_sheet)
            if current_liabilities > 0:
                result['Current_Ratio'] = current_assets / current_liabilities
        except:
            pass
        
        # Cash Flow Quality metrics
        # OCF/Net Profit Ratio
        try:
            ocf = get_metric('operating activities', cashflow)
            net_income = get_metric('net income', financials)
            if net_income > 0:
                result['OCF_Net_Profit_Ratio'] = (ocf / net_income) * 100
        except:
            pass
        
        # FCF Trend Score (1-5)
        try:
            fcf_values = []
            for i in range(min(5, len(cashflow.columns))):
                fcf = get_metric('free cash flow', cashflow.iloc[:, i])
                if fcf:
                    fcf_values.append(fcf)
            
            if len(fcf_values) >= 3:
                # Check if FCF is positive and growing
                positive_count = sum(1 for fcf in fcf_values if fcf > 0)
                growth_count = 0
                
                for i in range(1, len(fcf_values)):
                    if fcf_values[i] > fcf_values[i-1]:
                        growth_count += 1
                
                if positive_count == len(fcf_values) and growth_count >= len(fcf_values)-1:
                    result['FCF_Trend_Score'] = 5  # Positive and growing
                elif positive_count == len(fcf_values):
                    result['FCF_Trend_Score'] = 4  # Positive, stable
                elif positive_count >= len(fcf_values)/2:
                    result['FCF_Trend_Score'] = 3  # Mixed positive/negative
                elif positive_count > 0:
                    result['FCF_Trend_Score'] = 2  # Mostly negative
                else:
                    result['FCF_Trend_Score'] = 1  # Negative and worsening
        except:
            pass
        
        # Cash Flow Stability (1-5)
        try:
            ocf_values = []
            for i in range(min(5, len(cashflow.columns))):
                ocf = get_metric('operating activities', cashflow.iloc[:, i])
                if ocf:
                    ocf_values.append(ocf)
            
            if len(ocf_values) >= 3:
                avg_ocf = sum(ocf_values) / len(ocf_values)
                if avg_ocf > 0:
                    std_dev = np.std(ocf_values)
                    volatility = (std_dev / avg_ocf) * 100
                    
                    if volatility < 20:
                        result['Cash_Flow_Stability'] = 5
                    elif volatility < 35:
                        result['Cash_Flow_Stability'] = 4
                    elif volatility < 50:
                        result['Cash_Flow_Stability'] = 3
                    elif volatility < 70:
                        result['Cash_Flow_Stability'] = 2
                    else:
                        result['Cash_Flow_Stability'] = 1
        except:
            pass
        
        # OCF vs Debt Repayment Capacity (1-5)
        try:
            ocf = get_metric('operating activities', cashflow)
            total_debt = get_metric('total debt', balance_sheet)
            if total_debt > 0:
                ratio = ocf / total_debt
                if ratio >= 1.5:
                    result['OCF_Debt_Ratio'] = 5
                elif ratio >= 1.0:
                    result['OCF_Debt_Ratio'] = 4
                elif ratio >= 0.6:
                    result['OCF_Debt_Ratio'] = 3
                elif ratio >= 0.3:
                    result['OCF_Debt_Ratio'] = 2
                else:
                    result['OCF_Debt_Ratio'] = 1
        except:
            pass
        
        return result
    except Exception as e:
        st.error(f"Error calculating fundamentals for {data.get('ticker','')}: {str(e)}")
        return None

def calculate_industry_averages(fundamentals_list):
    if not fundamentals_list:
        return None
    df = pd.DataFrame(fundamentals_list)
    
    # Basic metrics
    industry_avgs = {
        'sales_cagr': df['Sales_CAGR_5Y'].mean() if 'Sales_CAGR_5Y' in df.columns and not df['Sales_CAGR_5Y'].isnull().all() else 0,
        'pat_cagr': df['PAT_CAGR_5Y'].mean() if 'PAT_CAGR_5Y' in df.columns and not df['PAT_CAGR_5Y'].isnull().all() else 0,
        'de': df['Debt_to_Equity'].mean() if 'Debt_to_Equity' in df.columns and not df['Debt_to_Equity'].isnull().all() else 0,
        'pe': df['PE_Ratio'].mean() if 'PE_Ratio' in df.columns and not df['PE_Ratio'].isnull().all() else 0,
        'roe': df['ROE'].mean() if 'ROE' in df.columns and not df['ROE'].isnull().all() else 0,
        'promoter_holding_trend': df['Promoter_Holding_Trend'].mean() if 'Promoter_Holding_Trend' in df.columns and not df['Promoter_Holding_Trend'].isnull().all() else 3,
        'pledged_shares': df['Pledged_Shares_Percent'].mean() if 'Pledged_Shares_Percent' in df.columns and not df['Pledged_Shares_Percent'].isnull().all() else 0,
        'board_audit_quality': df['Board_Audit_Quality'].mean() if 'Board_Audit_Quality' in df.columns and not df['Board_Audit_Quality'].isnull().all() else 3,
        'institutional_holding': df['Institutional_Holding'].mean() if 'Institutional_Holding' in df.columns and not df['Institutional_Holding'].isnull().all() else 0,
        # Earnings Quality
        'revenue_cagr_3y': df['Revenue_CAGR_3Y'].mean() if 'Revenue_CAGR_3Y' in df.columns and not df['Revenue_CAGR_3Y'].isnull().all() else 0,
        'operating_profit_cagr': df['Operating_Profit_CAGR'].mean() if 'Operating_Profit_CAGR' in df.columns and not df['Operating_Profit_CAGR'].isnull().all() else 0,
        'pat_cagr_3y': df['PAT_CAGR_3Y'].mean() if 'PAT_CAGR_3Y' in df.columns and not df['PAT_CAGR_3Y'].isnull().all() else 0,
        'eps_growth_yoy': df['EPS_Growth_YoY'].mean() if 'EPS_Growth_YoY' in df.columns and not df['EPS_Growth_YoY'].isnull().all() else 0,
        'sales_growth_yoy': df['Sales_Growth_YoY'].mean() if 'Sales_Growth_YoY' in df.columns and not df['Sales_Growth_YoY'].isnull().all() else 0,
        # Profitability
        'roce': df['ROCE'].mean() if 'ROCE' in df.columns and not df['ROCE'].isnull().all() else 0,
        'net_profit_margin': df['Net_Profit_Margin'].mean() if 'Net_Profit_Margin' in df.columns and not df['Net_Profit_Margin'].isnull().all() else 0,
        # Valuation
        'ev_ebitda': df['EV_EBITDA'].mean() if 'EV_EBITDA' in df.columns and not df['EV_EBITDA'].isnull().all() else 0,
        'price_to_book': df['Price_to_Book'].mean() if 'Price_to_Book' in df.columns and not df['Price_to_Book'].isnull().all() else 0,
        # Balance Sheet
        'interest_coverage': df['Interest_Coverage_Ratio'].mean() if 'Interest_Coverage_Ratio' in df.columns and not df['Interest_Coverage_Ratio'].isnull().all() else 0,
        'current_ratio': df['Current_Ratio'].mean() if 'Current_Ratio' in df.columns and not df['Current_Ratio'].isnull().all() else 0,
        # Cash Flow
        'ocf_net_profit_ratio': df['OCF_Net_Profit_Ratio'].mean() if 'OCF_Net_Profit_Ratio' in df.columns and not df['OCF_Net_Profit_Ratio'].isnull().all() else 0,
        'fcf_trend': df['FCF_Trend_Score'].mean() if 'FCF_Trend_Score' in df.columns and not df['FCF_Trend_Score'].isnull().all() else 0,
        'cash_flow_stability': df['Cash_Flow_Stability'].mean() if 'Cash_Flow_Stability' in df.columns and not df['Cash_Flow_Stability'].isnull().all() else 0,
        'ocf_debt_ratio': df['OCF_Debt_Ratio'].mean() if 'OCF_Debt_Ratio' in df.columns and not df['OCF_Debt_Ratio'].isnull().all() else 0
    }
    
    return industry_avgs

def calculate_detailed_scores(fund: dict, industry: dict):
    """
    Calculate detailed scores for each factor (1-5 scale) based on the provided scoring logic.
    Returns a dictionary with factor scores and the overall score.
    """
    scores = {}
    
    # 1. Management & Governance (20% weight)
    mg_scores = {}
    
    # Promoter Holding Trend: 1-5 scale
    promoter_trend = fund.get("Promoter_Holding_Trend", 3)
    mg_scores["Promoter_Holding_Trend"] = promoter_trend if promoter_trend is not None else 3
    
    # Pledged Shares %: 1-5 scale
    pledged = fund.get("Pledged_Shares_Percent", 0)
    pledged = pledged if pledged is not None else 0
    if pledged == 0:
        pledged_score = 5
    elif pledged < 5:
        pledged_score = 4
    elif pledged <= 15:
        pledged_score = 2
    else:
        pledged_score = 1
    mg_scores["Pledged_Shares_Percent"] = pledged_score
    
    # Board & Audit Quality: 1-5 scale
    board_quality = fund.get("Board_Audit_Quality", 3)
    mg_scores["Board_Audit_Quality"] = board_quality if board_quality is not None else 3
    
    # Institutional Holding: 1-5 scale
    institutional = fund.get("Institutional_Holding", 0)
    institutional = institutional if institutional is not None else 0
    if institutional > 20:
        institutional_score = 5
    elif 10 <= institutional <= 20:
        institutional_score = 3
    else:
        institutional_score = 1
    mg_scores["Institutional_Holding"] = institutional_score
    
    # Calculate Management & Governance factor score (average of sub-scores)
    mg_factor_score = sum(mg_scores.values()) / len(mg_scores)
    scores["Management_Governance"] = {
        "score": mg_factor_score,
        "sub_scores": mg_scores,
        "weight": 0.20
    }
    
    # 2. Earnings Quality / Growth (25% weight)
    eq_scores = {}
    
    # Revenue CAGR (3Y): 1-5 scale
    revenue_cagr = fund.get("Revenue_CAGR_3Y", 0)
    if revenue_cagr is None:
        revenue_cagr_score = 3
    elif revenue_cagr > 20:
        revenue_cagr_score = 5
    elif 10 <= revenue_cagr <= 20:
        revenue_cagr_score = 4
    elif 5 <= revenue_cagr < 10:
        revenue_cagr_score = 3
    else:
        revenue_cagr_score = 1
    eq_scores["Revenue_CAGR_3Y"] = revenue_cagr_score
    
    # Operating Profit CAGR: 1-5 scale
    operating_profit_cagr = fund.get("Operating_Profit_CAGR", 0)
    if operating_profit_cagr is None:
        operating_profit_cagr_score = 3
    elif operating_profit_cagr > 20:
        operating_profit_cagr_score = 5
    elif 10 <= operating_profit_cagr <= 20:
        operating_profit_cagr_score = 4
    elif 5 <= operating_profit_cagr < 10:
        operating_profit_cagr_score = 3
    else:
        operating_profit_cagr_score = 1
    eq_scores["Operating_Profit_CAGR"] = operating_profit_cagr_score
    
    # PAT CAGR (3Y): 1-5 scale
    pat_cagr = fund.get("PAT_CAGR_3Y", 0)
    if pat_cagr is None:
        pat_cagr_score = 3
    elif pat_cagr > 20:
        pat_cagr_score = 5
    elif 10 <= pat_cagr <= 20:
        pat_cagr_score = 4
    elif 5 <= pat_cagr < 10:
        pat_cagr_score = 3
    else:
        pat_cagr_score = 1
    eq_scores["PAT_CAGR_3Y"] = pat_cagr_score
    
    # EPS Growth vs Sales Growth: 1-5 scale
    eps_growth = fund.get("EPS_Growth_YoY", 0)
    sales_growth = fund.get("Sales_Growth_YoY", 0)
    
    if eps_growth is None or sales_growth is None:
        eps_vs_sales_score = 3
    elif eps_growth < 0:
        eps_vs_sales_score = 1
    elif eps_growth < sales_growth:
        eps_vs_sales_score = 2
    elif abs(eps_growth - sales_growth) <= 10:  # Within 10%
        eps_vs_sales_score = 3
    elif eps_growth > sales_growth + 25:  # EPS > Sales by 25-50%
        eps_vs_sales_score = 4
    else:  # EPS > Sales by >50%
        eps_vs_sales_score = 5
    eq_scores["EPS_vs_Sales_Growth"] = eps_vs_sales_score
    
    # Calculate Earnings Quality factor score (average of sub-scores)
    eq_factor_score = sum(eq_scores.values()) / len(eq_scores)
    scores["Earnings_Quality"] = {
        "score": eq_factor_score,
        "sub_scores": eq_scores,
        "weight": 0.25
    }
    
    # 3. Profitability (20% weight)
    prof_scores = {}
    
    # ROE: 1-5 scale (relative to industry)
    roe = fund.get("ROE", 0)
    industry_roe = industry.get("roe", 0)
    
    if roe is None:
        roe_score = 3
    elif roe >= industry_roe + 10:
        roe_score = 5
    elif roe >= industry_roe + 5:
        roe_score = 4
    elif abs(roe - industry_roe) <= 5:
        roe_score = 3
    elif roe >= industry_roe - 10:
        roe_score = 2
    else:
        roe_score = 1
    prof_scores["ROE"] = roe_score
    
    # ROCE: 1-5 scale (relative to industry)
    roce = fund.get("ROCE", 0)
    industry_roce = industry.get("roce", 0)
    
    if roce is None:
        roce_score = 3
    elif roce >= industry_roce + 10:
        roce_score = 5
    elif roce >= industry_roce + 5:
        roce_score = 4
    elif abs(roce - industry_roce) <= 5:
        roce_score = 3
    elif roce >= industry_roce - 10:
        roce_score = 2
    else:
        roce_score = 1
    prof_scores["ROCE"] = roce_score
    
    # Net Profit Margin: 1-5 scale (absolute)
    net_profit_margin = fund.get("Net_Profit_Margin", 0)
    if net_profit_margin is None:
        npm_score = 3
    elif net_profit_margin >= 20:
        npm_score = 5
    elif 15 <= net_profit_margin < 20:
        npm_score = 4
    elif 10 <= net_profit_margin < 15:
        npm_score = 3
    elif 5 <= net_profit_margin < 10:
        npm_score = 2
    else:
        npm_score = 1
    prof_scores["Net_Profit_Margin"] = npm_score
    
    # EPS Growth (YoY): 1-5 scale
    eps_growth = fund.get("EPS_Growth_YoY", 0)
    if eps_growth is None:
        eps_growth_score = 3
    elif eps_growth > 25:
        eps_growth_score = 5
    elif 15 <= eps_growth <= 25:
        eps_growth_score = 4
    elif 8 <= eps_growth < 15:
        eps_growth_score = 3
    elif 3 <= eps_growth < 8:
        eps_growth_score = 2
    else:
        eps_growth_score = 1
    prof_scores["EPS_Growth_YoY"] = eps_growth_score
    
    # Calculate Profitability factor score (average of sub-scores)
    prof_factor_score = sum(prof_scores.values()) / len(prof_scores)
    scores["Profitability"] = {
        "score": prof_factor_score,
        "sub_scores": prof_scores,
        "weight": 0.20
    }
    
    # 4. Valuation (15% weight)
    val_scores = {}
    
    # P/E: 1-5 scale (relative to industry)
    pe = fund.get("PE_Ratio", 0)
    industry_pe = industry.get("pe", 0)
    
    if pe is None or industry_pe is None or industry_pe <= 0:
        pe_score = 3
    else:
        pe_diff = ((pe - industry_pe) / industry_pe) * 100
        
        if pe_diff <= -25:
            pe_score = 5
        elif -25 < pe_diff <= -10:
            pe_score = 4
        elif -10 < pe_diff <= 10:
            pe_score = 3
        elif 10 < pe_diff <= 25:
            pe_score = 2
        else:
            pe_score = 1
    val_scores["PE_Ratio"] = pe_score
    
    # EV/EBITDA: 1-5 scale (relative to industry)
    ev_ebitda = fund.get("EV_EBITDA", 0)
    industry_ev_ebitda = industry.get("ev_ebitda", 0)
    
    if ev_ebitda is None or industry_ev_ebitda is None or industry_ev_ebitda <= 0:
        ev_ebitda_score = 3
    else:
        ev_ebitda_diff = ((ev_ebitda - industry_ev_ebitda) / industry_ev_ebitda) * 100
        
        if ev_ebitda_diff <= -20:
            ev_ebitda_score = 5
        elif -20 < ev_ebitda_diff <= -5:
            ev_ebitda_score = 4
        elif -5 < ev_ebitda_diff <= 5:
            ev_ebitda_score = 3
        elif 5 < ev_ebitda_diff <= 20:
            ev_ebitda_score = 2
        else:
            ev_ebitda_score = 1
    val_scores["EV_EBITDA"] = ev_ebitda_score
    
    # Price-to-Book: 1-5 scale (relative to industry)
    pb = fund.get("Price_to_Book", 0)
    industry_pb = industry.get("price_to_book", 0)
    
    if pb is None or industry_pb is None or industry_pb <= 0:
        pb_score = 3
    else:
        pb_diff = ((pb - industry_pb) / industry_pb) * 100
        
        if pb_diff <= -30:
            pb_score = 5
        elif -30 < pb_diff <= -10:
            pb_score = 4
        elif -10 < pb_diff <= 10:
            pb_score = 3
        elif 10 < pb_diff <= 30:
            pb_score = 2
        else:
            pb_score = 1
    val_scores["Price_to_Book"] = pb_score
    
    # Calculate Valuation factor score (average of sub-scores)
    val_factor_score = sum(val_scores.values()) / len(val_scores)
    scores["Valuation"] = {
        "score": val_factor_score,
        "sub_scores": val_scores,
        "weight": 0.15
    }
    
    # 5. Balance Sheet Quality/Solvency (10% weight)
    bs_scores = {}
    
    # Debt-to-Equity: 1-5 scale
    de = fund.get("Debt_to_Equity", 0)
    if de is None:
        de_score = 3
    elif de < 0.3:
        de_score = 5
    elif 0.3 <= de < 0.6:
        de_score = 4
    elif 0.6 <= de < 1.0:
        de_score = 3
    elif 1.0 <= de < 2.0:
        de_score = 2
    else:
        de_score = 1
    bs_scores["Debt_to_Equity"] = de_score
    
    # Interest Coverage Ratio: 1-5 scale
    icr = fund.get("Interest_Coverage_Ratio", 0)
    if icr is None:
        icr_score = 3
    elif icr >= 10:
        icr_score = 5
    elif 5 <= icr < 10:
        icr_score = 4
    elif 3 <= icr < 5:
        icr_score = 3
    elif 1.5 <= icr < 3:
        icr_score = 2
    else:
        icr_score = 1
    bs_scores["Interest_Coverage_Ratio"] = icr_score
    
    # Current Ratio: 1-5 scale
    current_ratio = fund.get("Current_Ratio", 0)
    if current_ratio is None:
        current_ratio_score = 3
    elif 1.5 <= current_ratio <= 3.0:
        current_ratio_score = 5
    elif 1.2 <= current_ratio < 1.5:
        current_ratio_score = 4
    elif 1.0 <= current_ratio < 1.2:
        current_ratio_score = 3
    elif 0.8 <= current_ratio < 1.0:
        current_ratio_score = 2
    else:
        current_ratio_score = 1
    bs_scores["Current_Ratio"] = current_ratio_score
    
    # Calculate Balance Sheet factor score (average of sub-scores)
    bs_factor_score = sum(bs_scores.values()) / len(bs_scores)
    scores["Balance_Sheet"] = {
        "score": bs_factor_score,
        "sub_scores": bs_scores,
        "weight": 0.10
    }
    
    # 6. Cash Flow Quality (10% weight)
    cf_scores = {}
    
    # OCF/Net Profit Ratio: 1-5 scale
    ocf_net_profit = fund.get("OCF_Net_Profit_Ratio", 0)
    if ocf_net_profit is None:
        ocf_net_profit_score = 3
    elif ocf_net_profit > 120:
        ocf_net_profit_score = 5
    elif 90 <= ocf_net_profit <= 120:
        ocf_net_profit_score = 4
    elif 70 <= ocf_net_profit < 90:
        ocf_net_profit_score = 3
    elif 50 <= ocf_net_profit < 70:
        ocf_net_profit_score = 2
    else:
        ocf_net_profit_score = 1
    cf_scores["OCF_Net_Profit_Ratio"] = ocf_net_profit_score
    
    # FCF Trend: 1-5 scale (already calculated)
    fcf_trend = fund.get("FCF_Trend_Score", 3)
    cf_scores["FCF_Trend"] = fcf_trend if fcf_trend is not None else 3
    
    # Cash Flow Stability: 1-5 scale (already calculated)
    cash_flow_stability = fund.get("Cash_Flow_Stability", 3)
    cf_scores["Cash_Flow_Stability"] = cash_flow_stability if cash_flow_stability is not None else 3
    
    # OCF vs Debt Repayment Capacity: 1-5 scale (already calculated)
    ocf_debt_ratio = fund.get("OCF_Debt_Ratio", 3)
    cf_scores["OCF_Debt_Ratio"] = ocf_debt_ratio if ocf_debt_ratio is not None else 3
    
    # Calculate Cash Flow factor score (average of sub-scores)
    cf_factor_score = sum(cf_scores.values()) / len(cf_scores)
    scores["Cash_Flow"] = {
        "score": cf_factor_score,
        "sub_scores": cf_scores,
        "weight": 0.10
    }
    
    # Calculate overall score (weighted average of factor scores)
    overall_score = 0
    for factor, data in scores.items():
        overall_score += data["score"] * data["weight"]
    
    # Determine recommendation based on overall score
    if overall_score >= 4.5:
        recommendation = "Strong Buy"
    elif overall_score >= 3.5:
        recommendation = "Buy"
    elif overall_score >= 2.5:
        recommendation = "Hold"
    else:
        recommendation = "Sell"
    
    return {
        "factor_scores": scores,
        "overall_score": overall_score,
        "recommendation": recommendation
    }

# ─── VISUALIZATION HELPER FUNCTIONS ───────────────────────────────────────────
def plot_historical_price(data, ticker):
    """Plot historical price and volume for a stock"""
    hist = data['hist']
    fig = make_subplots(
        rows=2, cols=1, 
        shared_xaxes=True, 
        vertical_spacing=0.1, 
        subplot_titles=(f'{ticker} Price', f'{ticker} Volume'),
        row_heights=[0.7, 0.3]
    )
    
    # Price chart
    fig.add_trace(
        go.Candlestick(
            x=hist.index,
            open=hist['Open'],
            high=hist['High'],
            low=hist['Low'],
            close=hist['Close'],
            name="Price"
        ),
        row=1, col=1
    )
    
    # Volume chart
    fig.add_trace(
        go.Bar(
            x=hist.index,
            y=hist['Volume'],
            name="Volume",
            marker_color='lightblue'
        ),
        row=2, col=1
    )
    
    # Add moving averages
    hist['MA20'] = hist['Close'].rolling(window=20).mean()
    hist['MA50'] = hist['Close'].rolling(window=50).mean()
    
    fig.add_trace(
        go.Scatter(
            x=hist.index,
            y=hist['MA20'],
            mode='lines',
            name='MA 20',
            line=dict(color='orange', width=1)
        ),
        row=1, col=1
    )
    
    fig.add_trace(
        go.Scatter(
            x=hist.index,
            y=hist['MA50'],
            mode='lines',
            name='MA 50',
            line=dict(color='blue', width=1)
        ),
        row=1, col=1
    )
    
    fig.update_layout(
        title=f'{ticker} Historical Price and Volume',
        xaxis_rangeslider_visible=False,
        height=600,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
    )
    
    return fig

def plot_financial_trends(data, ticker):
    """Plot financial metrics trends over time"""
    financials = data['financials']
    if financials.empty:
        return None
    
    # Get available metrics
    metrics = []
    values = []
    
    for metric in ['Total Revenue', 'Net Income', 'Operating Income', 'Gross Profit']:
        if metric in financials.index:
            metrics.append(metric)
            values.append(financials.loc[metric])
    
    if not metrics:
        return None
    
    fig = go.Figure()
    
    for i, metric in enumerate(metrics):
        fig.add_trace(
            go.Scatter(
                x=values[i].index,
                y=values[i].values,
                mode='lines+markers',
                name=metric,
                line=dict(width=2)
            )
        )
    
    fig.update_layout(
        title=f'{ticker} Financial Trends',
        xaxis_title='Date',
        yaxis_title='Amount',
        height=500,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
    )
    
    return fig

def plot_peer_comparison(fundamentals, industry, ticker):
    """Plot peer comparison for selected stock"""
    if ticker not in fundamentals.index:
        return None
    
    # Get selected stock data
    stock_data = fundamentals.loc[ticker]
    
    # Prepare comparison data
    metrics = [
        'Sales_CAGR_5Y', 'PAT_CAGR_5Y', 'ROE', 'PE_Ratio', 'Debt_to_Equity',
        'Revenue_CAGR_3Y', 'ROCE', 'Net_Profit_Margin', 'EV_EBITDA', 'Price_to_Book'
    ]
    
    # Extract values
    stock_values = []
    industry_values = []
    
    for metric in metrics:
        if metric in stock_data.index:
            # Convert to numeric if needed
            try:
                if isinstance(stock_data[metric], str):
                    val = float(stock_data[metric].replace('%', '').replace('x', ''))
                else:
                    val = float(stock_data[metric])
                stock_values.append(val)
            except:
                stock_values.append(0)
        else:
            stock_values.append(0)
        
        # Get industry value
        if metric == 'Sales_CAGR_5Y':
            industry_values.append(industry.get('sales_cagr', 0))
        elif metric == 'PAT_CAGR_5Y':
            industry_values.append(industry.get('pat_cagr', 0))
        elif metric == 'ROE':
            industry_values.append(industry.get('roe', 0))
        elif metric == 'PE_Ratio':
            industry_values.append(industry.get('pe', 0))
        elif metric == 'Debt_to_Equity':
            industry_values.append(industry.get('de', 0))
        elif metric == 'Revenue_CAGR_3Y':
            industry_values.append(industry.get('revenue_cagr_3y', 0))
        elif metric == 'ROCE':
            industry_values.append(industry.get('roce', 0))
        elif metric == 'Net_Profit_Margin':
            industry_values.append(industry.get('net_profit_margin', 0))
        elif metric == 'EV_EBITDA':
            industry_values.append(industry.get('ev_ebitda', 0))
        elif metric == 'Price_to_Book':
            industry_values.append(industry.get('price_to_book', 0))
        else:
            industry_values.append(0)
    
    # Create figure
    fig = go.Figure()
    
    # Add traces
    fig.add_trace(go.Bar(
        x=metrics,
        y=stock_values,
        name=ticker,
        marker_color='royalblue'
    ))
    
    fig.add_trace(go.Bar(
        x=metrics,
        y=industry_values,
        name='Industry Avg',
        marker_color='lightgray'
    ))
    
    fig.update_layout(
        title=f'{ticker} vs Industry Comparison',
        xaxis_title='Metrics',
        yaxis_title='Values',
        barmode='group',
        height=500
    )
    
    return fig

def plot_correlation_heatmap(fundamentals, ticker):
    """Plot correlation heatmap for financial metrics"""
    if ticker not in fundamentals.index:
        return None
    
    # Get selected stock data
    stock_data = fundamentals.loc[ticker]
    
    # Prepare data for correlation
    metrics = [
        'Sales_CAGR_5Y', 'PAT_CAGR_5Y', 'ROE', 'PE_Ratio', 'Debt_to_Equity',
        'Revenue_CAGR_3Y', 'ROCE', 'Net_Profit_Margin', 'EV_EBITDA', 'Price_to_Book'
    ]
    
    # Extract values
    values = []
    for metric in metrics:
        if metric in stock_data.index:
            try:
                if isinstance(stock_data[metric], str):
                    val = float(stock_data[metric].replace('%', '').replace('x', ''))
                else:
                    val = float(stock_data[metric])
                values.append(val)
            except:
                values.append(0)
        else:
            values.append(0)
    
    # Create DataFrame
    df = pd.DataFrame([values], columns=metrics)
    
    # Calculate correlation matrix
    corr_matrix = df.corr()
    
    # Create heatmap
    fig = go.Figure(data=go.Heatmap(
        z=corr_matrix.values,
        x=corr_matrix.columns,
        y=corr_matrix.columns,
        colorscale='RdBu',
        zmid=0,
        text=np.round(corr_matrix.values, 2),
        texttemplate="%{text}",
        textfont={"size": 10}
    ))
    
    fig.update_layout(
        title=f'{ticker} Financial Metrics Correlation',
        height=600
    )
    
    return fig

def plot_dupont_analysis(data, ticker):
    """Plot DuPont analysis for ROE decomposition"""
    financials = data['financials']
    balance_sheet = data['balance_sheet']
    
    if financials.empty or balance_sheet.empty:
        return None
    
    # Helper function to get metric
    def get_metric(name, df):
        matches = [idx for idx in df.index if name.lower() in idx.lower()]
        if matches:
            return df.loc[matches[0]].dropna()
        return pd.Series()
    
    # Get data
    net_income = get_metric('net income', financials)
    revenue = get_metric('revenue', financials)
    total_assets = get_metric('total assets', balance_sheet)
    equity = get_metric('total equity', balance_sheet)
    
    if net_income.empty or revenue.empty or total_assets.empty or equity.empty:
        return None
    
    # Calculate components
    # Net Profit Margin = Net Income / Revenue
    npm = (net_income / revenue).dropna()
    
    # Asset Turnover = Revenue / Total Assets
    at = (revenue / total_assets).dropna()
    
    # Financial Leverage = Total Assets / Equity
    fl = (total_assets / equity).dropna()
    
    # ROE = NPM * AT * FL
    roe = npm * at * fl
    
    # Create DataFrame for plotting
    dupont_df = pd.DataFrame({
        'Net Profit Margin': npm,
        'Asset Turnover': at,
        'Financial Leverage': fl,
        'ROE': roe
    })
    
    # Create figure
    fig = go.Figure()
    
    # Add traces
    fig.add_trace(go.Scatter(
        x=dupont_df.index,
        y=dupont_df['Net Profit Margin'],
        mode='lines+markers',
        name='Net Profit Margin',
        line=dict(width=2)
    ))
    
    fig.add_trace(go.Scatter(
        x=dupont_df.index,
        y=dupont_df['Asset Turnover'],
        mode='lines+markers',
        name='Asset Turnover',
        line=dict(width=2)
    ))
    
    fig.add_trace(go.Scatter(
        x=dupont_df.index,
        y=dupont_df['Financial Leverage'],
        mode='lines+markers',
        name='Financial Leverage',
        line=dict(width=2)
    ))
    
    fig.add_trace(go.Scatter(
        x=dupont_df.index,
        y=dupont_df['ROE'],
        mode='lines+markers',
        name='ROE',
        line=dict(width=3, dash='dash')
    ))
    
    fig.update_layout(
        title=f'{ticker} DuPont Analysis (ROE Decomposition)',
        xaxis_title='Date',
        yaxis_title='Value',
        height=500,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
    )
    
    return fig

def plot_pca_analysis(fundamentals_df):
    """Plot PCA analysis for dimensionality reduction"""
    # Select numeric columns
    numeric_cols = []
    for col in fundamentals_df.columns:
        try:
            # Try to convert to numeric
            pd.to_numeric(fundamentals_df[col], errors='raise')
            numeric_cols.append(col)
        except:
            pass
    
    if len(numeric_cols) < 3:
        return None
    
    # Prepare data
    df = fundamentals_df[numeric_cols].copy()
    
    # Convert to numeric
    for col in df.columns:
        df[col] = pd.to_numeric(df[col], errors='coerce')
    
    # Drop rows with NaN
    df = df.dropna()
    
    if len(df) < 2:
        return None
    
    # Standardize data
    scaler = MinMaxScaler()
    scaled_data = scaler.fit_transform(df)
    
    # Apply PCA
    pca = PCA(n_components=2)
    principal_components = pca.fit_transform(scaled_data)
    
    # Create DataFrame
    pca_df = pd.DataFrame(
        data=principal_components,
        columns=['PC1', 'PC2'],
        index=df.index
    )
    
    # Create scatter plot
    fig = px.scatter(
        pca_df,
        x='PC1',
        y='PC2',
        color=pca_df.index,
        title='PCA Analysis of Stocks',
        height=600
    )
    
    # Add explained variance ratio to the plot
    explained_var = pca.explained_variance_ratio_
    fig.update_layout(
        annotations=[
            dict(
                x=0.5,
                y=-0.1,
                xref='paper',
                yref='paper',
                text=f'Explained Variance: PC1 {explained_var[0]:.2%}, PC2 {explained_var[1]:.2%}',
                showarrow=False,
                font=dict(size=12)
            )
        ]
    )
    
    return fig

def plot_score_distribution(detailed_scores_list):
    """Plot distribution of overall scores"""
    scores = [item['overall_score'] for item in detailed_scores_list]
    tickers = [item['Ticker'] for item in detailed_scores_list]
    
    # Create DataFrame
    df = pd.DataFrame({
        'Ticker': tickers,
        'Score': scores
    })
    
    # Create bar chart
    fig = px.bar(
        df,
        x='Ticker',
        y='Score',
        color='Score',
        color_continuous_scale='RdYlGn',
        title='Stock Score Distribution',
        height=500
    )
    
    # Add horizontal lines for score thresholds
    fig.add_hline(y=4.5, line_dash="dash", line_color="green", annotation_text="Strong Buy")
    fig.add_hline(y=3.5, line_dash="dash", line_color="lightgreen", annotation_text="Buy")
    fig.add_hline(y=2.5, line_dash="dash", line_color="orange", annotation_text="Hold")
    
    return fig

def plot_factor_contributions(detailed_scores_list):
    """Plot factor contributions to overall score"""
    # Prepare data
    data = []
    for item in detailed_scores_list:
        for factor, factor_data in item['factor_scores'].items():
            data.append({
                'Ticker': item['Ticker'],
                'Factor': factor.replace('_', ' '),
                'Score': factor_data['score'],
                'Weight': factor_data['weight'],
                'Contribution': factor_data['score'] * factor_data['weight']
            })
    
    df = pd.DataFrame(data)
    
    # Create stacked bar chart
    fig = px.bar(
        df,
        x='Ticker',
        y='Contribution',
        color='Factor',
        title='Factor Contributions to Overall Score',
        height=600
    )
    
    return fig

# ─── MAIN APP ────────────────────────────────────────────────────────────────
st.set_page_config(page_title="Stock Analysis Dashboard", layout="wide")
st.markdown("""
<style>
.metric-card {
    background-color: var(--card-background);
    padding: 1rem;
    border-radius: 8px;
    box-shadow: 0 2px 4px rgba(0,0,0,0.1);
    margin-bottom: 1rem;
}
.subheader {
    font-weight: bold;
    margin-bottom: 0.5rem;
}
</style>
""", unsafe_allow_html=True)

st.title("📈 Stock Analysis Dashboard")

default_tickers = ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'TSLA', 'RELIANCE.BO', 'TCS.BO', 'INFY.BO', 'HDFCBANK.BO', 'SBIN.BO']

# Sidebar
with st.sidebar:
    st.markdown("## 🎛️ Control Panel")
    tickers = st.multiselect(
        "Select Stocks (max 10)",
        default_tickers,
        default=default_tickers[:3],
        help="Choose up to 10 stocks to compare"
    )
    st.markdown("### Management & Governance Inputs")
    with st.expander("Enter Promoter & Board Data (Optional)"):
        for ticker in tickers:
            st.markdown(f"#### {ticker}")
            promoter_trend = st.selectbox(
                f"Promoter Holding Trend ({ticker})",
                options=[5, 3, 1],
                format_func=lambda x: {5: "Increasing/Stable", 3: "Declining <10%", 1: "Sharp Decline"}[x],
                key=f"promoter_{ticker}"
            )
            pledged_shares = st.number_input(
                f"Pledged Shares % ({ticker})",
                min_value=0.0,
                max_value=100.0,
                value=0.0,
                step=0.1,
                key=f"pledged_{ticker}"
            )
            board_quality = st.selectbox(
                f"Board & Audit Quality ({ticker})",
                options=[5, 3, 1],
                format_func=lambda x: {5: "Strong", 3: "Moderate", 1: "Weak"}[x],
                key=f"board_{ticker}"
            )
    end_date = datetime.today()
    start_date = end_date - timedelta(days=20*365)
    st.caption(f"📅 Analysis period: {start_date.strftime('%Y-%m-%d')} to {end_date.strftime('%Y-%m-%d')}")

# Main app processing
if not tickers:
    st.warning("⚠️ Please select at least one stock to analyze.")
else:
    if len(tickers) > 10:
        st.error("❌ Please select no more than 10 stocks.")
        st.stop()
    
    with st.spinner("🔄 Fetching and analyzing data..."):
        progress_bar = st.progress(0)
        status_text = st.empty()
        all_data = []
        fundamentals_list = []
        detailed_scores_list = []
        
        for i, ticker in enumerate(tickers):
            status_text.text(f"📡 Fetching data for {ticker} ({i+1}/{len(tickers)})...")
            progress_bar.progress((i + 1) / len(tickers))
            data = get_stock_data(ticker)
            if data:
                all_data.append(data)
                fundamentals = calculate_fundamentals(data)
                if fundamentals:
                    fundamentals_list.append(fundamentals)
        
        industry_avgs = calculate_industry_averages(fundamentals_list)
        
        for fundamentals in fundamentals_list:
            detailed_scores = calculate_detailed_scores(fundamentals, industry_avgs)
            detailed_scores_list.append({
                'Ticker': fundamentals['Ticker'],
                **detailed_scores
            })
        
        status_text.text("✅ Analysis complete!")
        progress_bar.empty()
    
    if not fundamentals_list:
        st.error("❌ No data could be fetched for the selected tickers.")
        st.stop()
    
    # Create and format fundamentals DataFrame
    fundamentals_df = pd.DataFrame(fundamentals_list)
    fundamentals_df.set_index('Ticker', inplace=True)
    formatted_df = fundamentals_df.copy()
    
    for col in formatted_df.columns:
        if '%' in col and 'Ratio' not in col:
            formatted_df[col] = formatted_df[col].apply(lambda x: f"{x:.1f}%" if pd.notnull(x) else "N/A")
        elif 'Ratio' in col and col != 'Debt_to_Equity':
            formatted_df[col] = formatted_df[col].apply(lambda x: f"{x:.2f}x" if pd.notnull(x) else "N/A")
        elif 'Debt_to_Equity' in col:
            formatted_df[col] = formatted_df[col].apply(lambda x: f"{x:.2f}" if pd.notnull(x) else "N/A")
        elif 'Avg' in col or 'Latest' in col:
            formatted_df[col] = formatted_df[col].apply(
                lambda x: f"${x/1e9:.1f}B" if pd.notnull(x) and abs(x) >= 1e9 else
                          f"${x/1e6:.1f}M" if pd.notnull(x) and abs(x) >= 1e6 else
                          f"${x:.1f}" if pd.notnull(x) else "N/A"
            )
        elif col in ['Promoter_Holding_Trend', 'Board_Audit_Quality', 'FCF_Trend_Score', 'Cash_Flow_Stability', 'OCF_Debt_Ratio']:
            formatted_df[col] = formatted_df[col].apply(lambda x: f"{x}/5" if pd.notnull(x) else "N/A")
        elif col == 'Pledged_Shares_Percent':
            formatted_df[col] = formatted_df[col].apply(lambda x: f"{x:.1f}%" if pd.notnull(x) else "N/A")
        elif col == 'Institutional_Holding':
            formatted_df[col] = formatted_df[col].apply(lambda x: f"{x:.1f}%" if pd.notnull(x) else "N/A")
    
    # Update industry average row
    industry_row = {
        'Sales_CAGR_5Y': f"{industry_avgs['sales_cagr']:.1f}%" if pd.notnull(industry_avgs['sales_cagr']) else "N/A",
        'PAT_CAGR_5Y': f"{industry_avgs['pat_cagr']:.1f}%" if pd.notnull(industry_avgs['pat_cagr']) else "N/A",
        'Debt_to_Equity': f"{industry_avgs['de']:.2f}" if pd.notnull(industry_avgs['de']) else "N/A",
        'PE_Ratio': f"{industry_avgs['pe']:.1f}x" if pd.notnull(industry_avgs['pe']) else "N/A",
        'ROE': f"{industry_avgs['roe']:.1f}%" if pd.notnull(industry_avgs['roe']) else "N/A",
        'CFO_Latest': "N/A",
        'FCF_Latest': "N/A",
        'Revenue_Growth_YoY': "N/A",
        'Net_Income_Growth_YoY': "N/A",
        'CFO_5Y_Avg': "N/A",
        'FCF_5Y_Avg': "N/A",
        'Promoter_Holding_Trend': f"{industry_avgs['promoter_holding_trend']:.1f}/5" if pd.notnull(industry_avgs['promoter_holding_trend']) else "N/A",
        'Pledged_Shares_Percent': f"{industry_avgs['pledged_shares']:.1f}%" if pd.notnull(industry_avgs['pledged_shares']) else "N/A",
        'Board_Audit_Quality': f"{industry_avgs['board_audit_quality']:.1f}/5" if pd.notnull(industry_avgs['board_audit_quality']) else "N/A",
        'Institutional_Holding': f"{industry_avgs['institutional_holding']:.1f}%" if pd.notnull(industry_avgs['institutional_holding']) else "N/A",
        # Earnings Quality
        'Revenue_CAGR_3Y': f"{industry_avgs['revenue_cagr_3y']:.1f}%" if pd.notnull(industry_avgs['revenue_cagr_3y']) else "N/A",
        'Operating_Profit_CAGR': f"{industry_avgs['operating_profit_cagr']:.1f}%" if pd.notnull(industry_avgs['operating_profit_cagr']) else "N/A",
        'PAT_CAGR_3Y': f"{industry_avgs['pat_cagr_3y']:.1f}%" if pd.notnull(industry_avgs['pat_cagr_3y']) else "N/A",
        'EPS_Growth_YoY': f"{industry_avgs['eps_growth_yoy']:.1f}%" if pd.notnull(industry_avgs['eps_growth_yoy']) else "N/A",
        'Sales_Growth_YoY': f"{industry_avgs['sales_growth_yoy']:.1f}%" if pd.notnull(industry_avgs['sales_growth_yoy']) else "N/A",
        # Profitability
        'ROCE': f"{industry_avgs['roce']:.1f}%" if pd.notnull(industry_avgs['roce']) else "N/A",
        'Net_Profit_Margin': f"{industry_avgs['net_profit_margin']:.1f}%" if pd.notnull(industry_avgs['net_profit_margin']) else "N/A",
        # Valuation
        'EV_EBITDA': f"{industry_avgs['ev_ebitda']:.2f}x" if pd.notnull(industry_avgs['ev_ebitda']) else "N/A",
        'Price_to_Book': f"{industry_avgs['price_to_book']:.2f}x" if pd.notnull(industry_avgs['price_to_book']) else "N/A",
        # Balance Sheet
        'Interest_Coverage_Ratio': f"{industry_avgs['interest_coverage']:.1f}x" if pd.notnull(industry_avgs['interest_coverage']) else "N/A",
        'Current_Ratio': f"{industry_avgs['current_ratio']:.2f}x" if pd.notnull(industry_avgs['current_ratio']) else "N/A",
        # Cash Flow
        'OCF_Net_Profit_Ratio': f"{industry_avgs['ocf_net_profit_ratio']:.1f}%" if pd.notnull(industry_avgs['ocf_net_profit_ratio']) else "N/A",
        'FCF_Trend_Score': f"{industry_avgs['fcf_trend']:.1f}/5" if pd.notnull(industry_avgs['fcf_trend']) else "N/A",
        'Cash_Flow_Stability': f"{industry_avgs['cash_flow_stability']:.1f}/5" if pd.notnull(industry_avgs['cash_flow_stability']) else "N/A",
        'OCF_Debt_Ratio': f"{industry_avgs['ocf_debt_ratio']:.1f}/5" if pd.notnull(industry_avgs['ocf_debt_ratio']) else "N/A"
    }
    
    formatted_df.loc['Industry Avg'] = industry_row
    
    # Update sidebar with new metrics
    with st.sidebar:
        st.markdown("## 📊 Industry Benchmarks")
        st.markdown(f"""
        <div class="metric-card">
            <div>📈 <strong>Sales CAGR (5Y):</strong> {industry_row['Sales_CAGR_5Y']}</div>
            <div>💰 <strong>PAT CAGR (5Y):</strong> {industry_row['PAT_CAGR_5Y']}</div>
            <div>🏦 <strong>Debt-to-Equity:</strong> {industry_row['Debt_to_Equity']}</div>
            <div>🔍 <strong>P/E Ratio:</strong> {industry_row['PE_Ratio']}</div>
            <div>📊 <strong>ROE:</strong> {industry_row['ROE']}</div>
            <div>👥 <strong>Promoter Holding Trend:</strong> {industry_row['Promoter_Holding_Trend']}</div>
            <div>🔒 <strong>Pledged Shares:</strong> {industry_row['Pledged_Shares_Percent']}</div>
            <div>🏛️ <strong>Board & Audit Quality:</strong> {industry_row['Board_Audit_Quality']}</div>
            <div>🏦 <strong>Institutional Holding:</strong> {industry_row['Institutional_Holding']}</div>
        </div>
        """, unsafe_allow_html=True)
    
    # Tabs
    tab1, tab2, tab3, tab4, tab5, tab6, tab7 = st.tabs([
        "📊 Overview",
        "📈 Fundamentals",
        "🏆 Scores",
        "📉 Charts",
        "🔍 Details",
        "📊 Factor Analysis",
        "🧠 Advanced Analytics"
    ])
    
    with tab1:  # Overview tab
        st.markdown("## 🏦 Company Overview")
        for ticker in tickers:
            if ticker in fundamentals_df.index:
                data = formatted_df.loc[ticker]
                scores = next((item for item in detailed_scores_list if item['Ticker'] == ticker), None)
                with st.expander(f"🔍 {ticker} - {scores['recommendation'] if scores else ''}", expanded=False):
                    cols = st.columns(4)
                    with cols[0]:
                        st.markdown(f"""
                        <div class="metric-card">
                            <div class="subheader">Valuation</div>
                            <div>P/E: {data['PE_Ratio']}</div>
                            <div>Industry: {industry_row['PE_Ratio']}</div>
                            <div>Score: {scores['factor_scores']['Valuation']['score']:.1f}/5</div>
                        </div>
                        """, unsafe_allow_html=True)
                    with cols[1]:
                        st.markdown(f"""
                        <div class="metric-card">
                            <div class="subheader">Growth</div>
                            <div>Sales CAGR: {data['Sales_CAGR_5Y']}</div>
                            <div>PAT CAGR: {data['PAT_CAGR_5Y']}</div>
                            <div>Score: {scores['factor_scores']['Earnings_Quality']['score']:.1f}/5</div>
                        </div>
                        """, unsafe_allow_html=True)
                    with cols[2]:
                        st.markdown(f"""
                        <div class="metric-card">
                            <div class="subheader">Profitability</div>
                            <div>ROE: {data['ROE']}</div>
                            <div>Industry: {industry_row['ROE']}</div>
                            <div>Score: {scores['factor_scores']['Profitability']['score']:.1f}/5</div>
                        </div>
                        """, unsafe_allow_html=True)
                    with cols[3]:
                        st.markdown(f"""
                        <div class="metric-card">
                            <div class="subheader">Management & Governance</div>
                            <div>Promoter Trend: {data['Promoter_Holding_Trend']}</div>
                            <div>Institutional: {data['Institutional_Holding']}</div>
                            <div>Score: {scores['factor_scores']['Management_Governance']['score']:.1f}/5</div>
                        </div>
                        """, unsafe_allow_html=True)
        
        st.markdown("## ⚡ Quick Comparison")
        quick_metrics = st.multiselect(
            "Select metrics to compare",
            options=['Sales_CAGR_5Y', 'PAT_CAGR_5Y', 'ROE', 'PE_Ratio', 'Debt_to_Equity',
                     'Promoter_Holding_Trend', 'Pledged_Shares_Percent', 'Board_Audit_Quality', 'Institutional_Holding',
                     'Revenue_CAGR_3Y', 'Operating_Profit_CAGR', 'PAT_CAGR_3Y', 'EPS_Growth_YoY',
                     'ROCE', 'Net_Profit_Margin', 'EV_EBITDA', 'Price_to_Book',
                     'Interest_Coverage_Ratio', 'Current_Ratio', 'OCF_Net_Profit_Ratio', 'FCF_Trend_Score'],
            default=['Sales_CAGR_5Y', 'PE_Ratio', 'Institutional_Holding']
        )
        
        if quick_metrics:
            fig = make_subplots(rows=1, cols=len(quick_metrics), subplot_titles=quick_metrics)
            for i, metric in enumerate(quick_metrics):
                values = fundamentals_df[metric].copy()
                if metric in ['Sales_CAGR_5Y', 'PAT_CAGR_5Y', 'ROE', 'PE_Ratio', 'Debt_to_Equity', 
                             'Promoter_Holding_Trend', 'Pledged_Shares_Percent', 'Board_Audit_Quality', 
                             'Institutional_Holding', 'Revenue_CAGR_3Y', 'Operating_Profit_CAGR', 
                             'PAT_CAGR_3Y', 'EPS_Growth_YoY', 'ROCE', 'Net_Profit_Margin', 
                             'EV_EBITDA', 'Price_to_Book', 'Interest_Coverage_Ratio', 'Current_Ratio', 
                             'OCF_Net_Profit_Ratio', 'FCF_Trend_Score']:
                    values = values.apply(
                        lambda x: float(x.replace('%', '').replace('/5', '').replace('x', '')) if pd.notnull(x) and isinstance(x, str) else float(x) if pd.notnull(x) else np.nan
                    )
                fig.add_trace(
                    go.Bar(
                        x=fundamentals_df.index,
                        y=values,
                        name=metric,
                        marker_color=px.colors.qualitative.Plotly[i % len(px.colors.qualitative.Plotly)]
                    ),
                    row=1, col=i+1
                )
                # Add industry average line
                if metric == 'Sales_CAGR_5Y':
                    industry_val = industry_avgs['sales_cagr']
                elif metric == 'PAT_CAGR_5Y':
                    industry_val = industry_avgs['pat_cagr']
                elif metric == 'ROE':
                    industry_val = industry_avgs['roe']
                elif metric == 'PE_Ratio':
                    industry_val = industry_avgs['pe']
                elif metric == 'Debt_to_Equity':
                    industry_val = industry_avgs['de']
                elif metric == 'Promoter_Holding_Trend':
                    industry_val = industry_avgs['promoter_holding_trend']
                elif metric == 'Pledged_Shares_Percent':
                    industry_val = industry_avgs['pledged_shares']
                elif metric == 'Board_Audit_Quality':
                    industry_val = industry_avgs['board_audit_quality']
                elif metric == 'Institutional_Holding':
                    industry_val = industry_avgs['institutional_holding']
                elif metric == 'Revenue_CAGR_3Y':
                    industry_val = industry_avgs['revenue_cagr_3y']
                elif metric == 'Operating_Profit_CAGR':
                    industry_val = industry_avgs['operating_profit_cagr']
                elif metric == 'PAT_CAGR_3Y':
                    industry_val = industry_avgs['pat_cagr_3y']
                elif metric == 'EPS_Growth_YoY':
                    industry_val = industry_avgs['eps_growth_yoy']
                elif metric == 'ROCE':
                    industry_val = industry_avgs['roce']
                elif metric == 'Net_Profit_Margin':
                    industry_val = industry_avgs['net_profit_margin']
                elif metric == 'EV_EBITDA':
                    industry_val = industry_avgs['ev_ebitda']
                elif metric == 'Price_to_Book':
                    industry_val = industry_avgs['price_to_book']
                elif metric == 'Interest_Coverage_Ratio':
                    industry_val = industry_avgs['interest_coverage']
                elif metric == 'Current_Ratio':
                    industry_val = industry_avgs['current_ratio']
                elif metric == 'OCF_Net_Profit_Ratio':
                    industry_val = industry_avgs['ocf_net_profit_ratio']
                elif metric == 'FCF_Trend_Score':
                    industry_val = industry_avgs['fcf_trend']
                else:
                    industry_val = None
                
                if pd.notnull(industry_val):
                    fig.add_hline(
                        y=industry_val,
                        line_dash="dash",
                        line_color="red",
                        row=1, col=i+1
                    )
            
            fig.update_layout(height=400, showlegend=False, margin=dict(l=50, r=50, b=50, t=50))
            st.plotly_chart(fig, use_container_width=True)
    
    with tab2:  # Fundamentals tab
        st.markdown("## 📈 Fundamental Metrics")
        st.dataframe(
            formatted_df.style.applymap(
                lambda x: 'color: green' if ('%' in str(x) and float(x.replace('%','')) > 0) else
                          ('color: red' if '%' in str(x) and float(x.replace('%','')) < 0 else ''),
                subset=[col for col in formatted_df.columns if '%' in col and 'Ratio' not in col]
            ),
            height=600
        )
        
        with st.expander("📖 Metric Explanations"):
            st.markdown("""
            **Financial Metrics:**
            - **Sales CAGR (5Y):** Compound annual growth rate of revenue over 5 years
            - **PAT CAGR (5Y):** Compound annual growth rate of profit after tax over 5 years
            - **Debt-to-Equity:** Total debt divided by shareholder equity
            - **P/E Ratio:** Price-to-Earnings ratio
            - **ROE %:** Return on Equity
            - **CFO:** Cash Flow from Operations (latest)
            - **FCF:** Free Cash Flow (latest)
            
            **Earnings Quality / Growth:**
            - **Revenue CAGR (3Y):** Compound annual growth rate of revenue over 3 years
            - **Operating Profit CAGR:** Compound annual growth rate of operating profit
            - **PAT CAGR (3Y):** Compound annual growth rate of profit after tax over 3 years
            - **EPS Growth (YoY):** Year-over-year growth in earnings per share
            - **Sales Growth (YoY):** Year-over-year growth in revenue
            
            **Profitability:**
            - **ROCE %:** Return on Capital Employed
            - **Net Profit Margin %:** Net profit as a percentage of revenue
            
            **Valuation:**
            - **EV/EBITDA:** Enterprise Value to Earnings Before Interest, Taxes, Depreciation, and Amortization
            - **Price-to-Book:** Market price per share divided by book value per share
            
            **Balance Sheet Quality:**
            - **Interest Coverage Ratio:** EBIT divided by interest expense
            - **Current Ratio:** Current assets divided by current liabilities
            
            **Cash Flow Quality:**
            - **OCF/Net Profit Ratio:** Operating cash flow as a percentage of net profit
            - **FCF Trend Score:** Trend of free cash flow over time (1-5 scale)
            - **Cash Flow Stability:** Stability of operating cash flow over time (1-5 scale)
            - **OCF/Debt Ratio:** Operating cash flow as a multiple of total debt (1-5 scale)
            
            **Management & Governance:**
            - **Promoter Holding Trend:** Stability of promoter ownership (5=Increasing/Stable, 3=Declining <10%, 1=Sharp Decline)
            - **Pledged Shares %:** Percentage of shares pledged by promoters
            - **Board & Audit Quality:** Quality of board independence and audits (5=Strong, 3=Moderate, 1=Weak)
            - **Institutional Holding:** Percentage of shares held by FII/DII
            """)
    
    with tab3:  # Scores tab
        st.markdown("## 🏆 Stock Scores & Recommendations")
        scores_data = []
        for item in detailed_scores_list:
            scores_data.append({
                'Ticker': item['Ticker'],
                'Management & Governance': item['factor_scores']['Management_Governance']['score'],
                'Earnings Quality': item['factor_scores']['Earnings_Quality']['score'],
                'Profitability': item['factor_scores']['Profitability']['score'],
                'Valuation': item['factor_scores']['Valuation']['score'],
                'Balance Sheet': item['factor_scores']['Balance_Sheet']['score'],
                'Cash Flow': item['factor_scores']['Cash_Flow']['score'],
                'Overall Score': item['overall_score'],
                'Recommendation': item['recommendation']
            })
        
        scores_df = pd.DataFrame(scores_data)
        scores_df.set_index('Ticker', inplace=True)
        
        def color_score(val):
            if val >= 4:
                return 'background-color: rgba(46, 125, 50, 0.2);'
            elif val >= 3:
                return 'background-color: rgba(255, 243, 224, 0.5);'
            else:
                return 'background-color: rgba(239, 154, 154, 0.2);'
        
        def color_recommendation(val):
            if val == 'Strong Buy':
                return 'background-color: rgba(46, 125, 50, 0.3); font-weight: bold;'
            elif val == 'Buy':
                return 'background-color: rgba(139, 195, 74, 0.3); font-weight: bold;'
            elif val == 'Hold':
                return 'background-color: rgba(255, 235, 59, 0.3); font-weight: bold;'
            else:
                return 'background-color: rgba(239, 154, 154, 0.3); font-weight: bold;'
        
        styled_scores = scores_df.style\
            .applymap(color_score, subset=['Management & Governance', 'Earnings Quality', 'Profitability', 'Valuation', 'Balance Sheet', 'Cash Flow', 'Overall Score'])\
            .applymap(color_recommendation, subset=['Recommendation'])
        
        st.dataframe(styled_scores, height=600)
        
        with st.expander("🔍 Score Interpretation Guide"):
            st.markdown("""
            **Scoring System (1-5 scale for each factor):**
            - **Management & Governance (20% weight):** Evaluates promoter stability, pledged shares, board quality, and institutional holding
            - **Earnings Quality / Growth (25% weight):** Assesses revenue and profit growth trends and EPS vs sales growth
            - **Profitability (20% weight):** Measures ROE, ROCE, net profit margin, and EPS growth
            - **Valuation (15% weight):** Compares P/E, EV/EBITDA, and P/B ratios to industry averages
            - **Balance Sheet Quality (10% weight):** Evaluates debt levels, interest coverage, and liquidity
            - **Cash Flow Quality (10% weight):** Assesses cash flow generation, stability, and debt repayment capacity
            
            **Overall Score Calculation:**
            - Weighted average of all factor scores
            - **Recommendation Thresholds:**
              - 🟢 **4.5-5.0:** Strong Buy
              - 🟡 **3.5-4.4:** Buy
              - 🟠 **2.5-3.4:** Hold
              - 🔴 **0.0-2.4:** Sell
            """)
    
    with tab4:  # Charts tab
        st.markdown("## 📊 Advanced Visualizations")
        
        # Factor Scores Radar Chart
        st.markdown("### 🎯 Factor Scores Comparison")
        selected_ticker = st.selectbox("Select stock for radar chart", tickers)
        
        if selected_ticker:
            scores_data = next((item for item in detailed_scores_list if item['Ticker'] == selected_ticker), None)
            if scores_data:
                categories = ['Management & Governance', 'Earnings Quality', 'Profitability', 'Valuation', 'Balance Sheet', 'Cash Flow']
                values = [
                    scores_data['factor_scores']['Management_Governance']['score'],
                    scores_data['factor_scores']['Earnings_Quality']['score'],
                    scores_data['factor_scores']['Profitability']['score'],
                    scores_data['factor_scores']['Valuation']['score'],
                    scores_data['factor_scores']['Balance_Sheet']['score'],
                    scores_data['factor_scores']['Cash_Flow']['score']
                ]
                
                fig = go.Figure()
                fig.add_trace(go.Scatterpolar(
                    r=values,
                    theta=categories,
                    fill='toself',
                    name=selected_ticker
                ))
                
                fig.update_layout(
                    polar=dict(
                        radialaxis=dict(
                            visible=True,
                            range=[0, 5]
                        )),
                    showlegend=True,
                    title=f"{selected_ticker} - Factor Scores"
                )
                
                st.plotly_chart(fig, use_container_width=True)
        
        # Historical Price and Volume Chart
        st.markdown("### 📈 Historical Price and Volume")
        selected_ticker = st.selectbox("Select stock for price chart", tickers, key="price_chart")
        
        if selected_ticker:
            data = next((item for item in all_data if item['ticker'] == selected_ticker), None)
            if data:
                fig = plot_historical_price(data, selected_ticker)
                st.plotly_chart(fig, use_container_width=True)
        
        # Financial Metrics Trend
        st.markdown("### 📊 Financial Metrics Trend")
        selected_ticker = st.selectbox("Select stock for financial trend", tickers, key="trend_chart")
        
        if selected_ticker:
            data = next((item for item in all_data if item['ticker'] == selected_ticker), None)
            if data:
                fig = plot_financial_trends(data, selected_ticker)
                if fig:
                    st.plotly_chart(fig, use_container_width=True)
                else:
                    st.warning("Insufficient data to plot financial trends")
        
        # Peer Comparison
        st.markdown("### 🆚 Peer Comparison")
        selected_ticker = st.selectbox("Select stock for peer comparison", tickers, key="peer_chart")
        
        if selected_ticker:
            fig = plot_peer_comparison(formatted_df, industry_avgs, selected_ticker)
            if fig:
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.warning("Insufficient data to plot peer comparison")
        
        # Correlation Heatmap
        st.markdown("### 🔗 Correlation Heatmap")
        selected_ticker = st.selectbox("Select stock for correlation heatmap", tickers, key="corr_chart")
        
        if selected_ticker:
            fig = plot_correlation_heatmap(formatted_df, selected_ticker)
            if fig:
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.warning("Insufficient data to plot correlation heatmap")
        
        # DuPont Analysis
        st.markdown("### 🔍 DuPont Analysis")
        selected_ticker = st.selectbox("Select stock for DuPont analysis", tickers, key="dupont_chart")
        
        if selected_ticker:
            data = next((item for item in all_data if item['ticker'] == selected_ticker), None)
            if data:
                fig = plot_dupont_analysis(data, selected_ticker)
                if fig:
                    st.plotly_chart(fig, use_container_width=True)
                else:
                    st.warning("Insufficient data to plot DuPont analysis")
    
    with tab5:  # Details tab
        st.markdown("## 🔍 Detailed Analysis")
        selected_ticker = st.selectbox("Select stock for detailed analysis", tickers)
        
        if selected_ticker and selected_ticker in formatted_df.index:
            st.markdown(f"### 📊 {selected_ticker} - Detailed Fundamental Metrics")
            
            # Get scores data for selected ticker
            scores_data = next((item for item in detailed_scores_list if item['Ticker'] == selected_ticker), None)
            
            # Display fundamental metrics
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                st.markdown("#### Growth Metrics")
                st.metric("Sales CAGR (5Y)", formatted_df.loc[selected_ticker, 'Sales_CAGR_5Y'])
                st.metric("PAT CAGR (5Y)", formatted_df.loc[selected_ticker, 'PAT_CAGR_5Y'])
                st.metric("Revenue CAGR (3Y)", formatted_df.loc[selected_ticker, 'Revenue_CAGR_3Y'])
                st.metric("Operating Profit CAGR", formatted_df.loc[selected_ticker, 'Operating_Profit_CAGR'])
                st.metric("PAT CAGR (3Y)", formatted_df.loc[selected_ticker, 'PAT_CAGR_3Y'])
                st.metric("EPS Growth (YoY)", formatted_df.loc[selected_ticker, 'EPS_Growth_YoY'])
                st.metric("Sales Growth (YoY)", formatted_df.loc[selected_ticker, 'Sales_Growth_YoY'])
            
            with col2:
                st.markdown("#### Profitability Metrics")
                st.metric("ROE", formatted_df.loc[selected_ticker, 'ROE'])
                st.metric("ROCE", formatted_df.loc[selected_ticker, 'ROCE'])
                st.metric("Net Profit Margin", formatted_df.loc[selected_ticker, 'Net_Profit_Margin'])
                st.metric("EPS Growth (YoY)", formatted_df.loc[selected_ticker, 'EPS_Growth_YoY'])
            
            with col3:
                st.markdown("#### Valuation Metrics")
                st.metric("P/E Ratio", formatted_df.loc[selected_ticker, 'PE_Ratio'])
                st.metric("EV/EBITDA", formatted_df.loc[selected_ticker, 'EV_EBITDA'])
                st.metric("Price-to-Book", formatted_df.loc[selected_ticker, 'Price_to_Book'])
            
            with col4:
                st.markdown("#### Balance Sheet Metrics")
                st.metric("Debt-to-Equity", formatted_df.loc[selected_ticker, 'Debt_to_Equity'])
                st.metric("Interest Coverage Ratio", formatted_df.loc[selected_ticker, 'Interest_Coverage_Ratio'])
                st.metric("Current Ratio", formatted_df.loc[selected_ticker, 'Current_Ratio'])
            
            st.markdown("#### Cash Flow Metrics")
            col5, col6, col7, col8 = st.columns(4)
            with col5:
                st.metric("Operating Cash Flow", formatted_df.loc[selected_ticker, 'CFO_Latest'])
            with col6:
                st.metric("Free Cash Flow", formatted_df.loc[selected_ticker, 'FCF_Latest'])
            with col7:
                st.metric("OCF/Net Profit Ratio", formatted_df.loc[selected_ticker, 'OCF_Net_Profit_Ratio'])
            with col8:
                st.metric("FCF Trend Score", formatted_df.loc[selected_ticker, 'FCF_Trend_Score'])
            
            st.markdown("#### Management & Governance")
            col9, col10, col11, col12 = st.columns(4)
            with col9:
                st.metric("Promoter Holding Trend", formatted_df.loc[selected_ticker, 'Promoter_Holding_Trend'])
            with col10:
                st.metric("Pledged Shares %", formatted_df.loc[selected_ticker, 'Pledged_Shares_Percent'])
            with col11:
                st.metric("Board & Audit Quality", formatted_df.loc[selected_ticker, 'Board_Audit_Quality'])
            with col12:
                st.metric("Institutional Holding", formatted_df.loc[selected_ticker, 'Institutional_Holding'])
            
            # Display factor scores
            if scores_data:
                st.markdown("### 🏆 Factor Scores Breakdown")
                
                # Create a DataFrame for factor scores
                factor_scores_df = pd.DataFrame({
                    'Factor': ['Management & Governance', 'Earnings Quality', 'Profitability', 'Valuation', 'Balance Sheet', 'Cash Flow'],
                    'Score': [
                        scores_data['factor_scores']['Management_Governance']['score'],
                        scores_data['factor_scores']['Earnings_Quality']['score'],
                        scores_data['factor_scores']['Profitability']['score'],
                        scores_data['factor_scores']['Valuation']['score'],
                        scores_data['factor_scores']['Balance_Sheet']['score'],
                        scores_data['factor_scores']['Cash_Flow']['score']
                    ],
                    'Weight': [0.20, 0.25, 0.20, 0.15, 0.10, 0.10],
                    'Weighted Score': [
                        scores_data['factor_scores']['Management_Governance']['score'] * 0.20,
                        scores_data['factor_scores']['Earnings_Quality']['score'] * 0.25,
                        scores_data['factor_scores']['Profitability']['score'] * 0.20,
                        scores_data['factor_scores']['Valuation']['score'] * 0.15,
                        scores_data['factor_scores']['Balance_Sheet']['score'] * 0.10,
                        scores_data['factor_scores']['Cash_Flow']['score'] * 0.10
                    ]
                })
                
                st.dataframe(factor_scores_df)
                
                st.markdown(f"### 🎯 Overall Recommendation: **{scores_data['recommendation']}**")
                st.progress(scores_data['overall_score'] / 5)
                st.markdown(f"**Overall Score:** {scores_data['overall_score']:.2f}/5.00")
                
                # Display sub-scores for each factor
                st.markdown("### 🔍 Sub-Scores Breakdown")
                
                # Management & Governance
                st.markdown("#### Management & Governance")
                mg_sub_scores = scores_data['factor_scores']['Management_Governance']['sub_scores']
                mg_df = pd.DataFrame({
                    'Metric': list(mg_sub_scores.keys()),
                    'Score': list(mg_sub_scores.values())
                })
                st.dataframe(mg_df)
                
                # Earnings Quality
                st.markdown("#### Earnings Quality")
                eq_sub_scores = scores_data['factor_scores']['Earnings_Quality']['sub_scores']
                eq_df = pd.DataFrame({
                    'Metric': list(eq_sub_scores.keys()),
                    'Score': list(eq_sub_scores.values())
                })
                st.dataframe(eq_df)
                
                # Profitability
                st.markdown("#### Profitability")
                prof_sub_scores = scores_data['factor_scores']['Profitability']['sub_scores']
                prof_df = pd.DataFrame({
                    'Metric': list(prof_sub_scores.keys()),
                    'Score': list(prof_sub_scores.values())
                })
                st.dataframe(prof_df)
                
                # Valuation
                st.markdown("#### Valuation")
                val_sub_scores = scores_data['factor_scores']['Valuation']['sub_scores']
                val_df = pd.DataFrame({
                    'Metric': list(val_sub_scores.keys()),
                    'Score': list(val_sub_scores.values())
                })
                st.dataframe(val_df)
                
                # Balance Sheet
                st.markdown("#### Balance Sheet Quality")
                bs_sub_scores = scores_data['factor_scores']['Balance_Sheet']['sub_scores']
                bs_df = pd.DataFrame({
                    'Metric': list(bs_sub_scores.keys()),
                    'Score': list(bs_sub_scores.values())
                })
                st.dataframe(bs_df)
                
                # Cash Flow
                st.markdown("#### Cash Flow Quality")
                cf_sub_scores = scores_data['factor_scores']['Cash_Flow']['sub_scores']
                cf_df = pd.DataFrame({
                    'Metric': list(cf_sub_scores.keys()),
                    'Score': list(cf_sub_scores.values())
                })
                st.dataframe(cf_df)
    
    with tab6:  # Factor Analysis tab
        st.markdown("## 📊 Factor Analysis")
        
        # Factor Scores Comparison
        st.markdown("### 🎯 Factor Scores Comparison")
        
        # Prepare data for factor comparison
        factor_comparison_data = []
        for item in detailed_scores_list:
            for factor, data in item['factor_scores'].items():
                # Ensure score and weight are scalars
                score = data['score']
                weight = data['weight']
                if isinstance(score, (list, dict, tuple)):
                    score = 3.0  # default
                if isinstance(weight, (list, dict, tuple)):
                    weight = 0.2  # default
                factor_comparison_data.append({
                    'Ticker': item['Ticker'],
                    'Factor': factor.replace('_', ' '),
                    'Score': score,
                    'Weight': weight
                })
        
        factor_comparison_df = pd.DataFrame(factor_comparison_data)
        
        # Create a bar chart for factor comparison
        fig = px.bar(
            factor_comparison_df,
            x='Factor',
            y='Score',
            color='Ticker',
            barmode='group',
            title="Factor Scores Comparison",
            color_discrete_sequence=px.colors.qualitative.Plotly
        )
        
        fig.update_layout(
            xaxis_title="Factor",
            yaxis_title="Score (1-5)",
            legend_title="Ticker",
            height=500
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
        # Weighted Contribution Analysis
        st.markdown("### ⚖️ Weighted Contribution Analysis")
        
        # Prepare data for weighted contribution
        weighted_contrib_data = []
        for item in detailed_scores_list:
            for factor, data in item['factor_scores'].items():
                # Ensure score and weight are scalars
                score = data['score']
                weight = data['weight']
                if isinstance(score, (list, dict, tuple)):
                    score = 3.0  # default
                if isinstance(weight, (list, dict, tuple)):
                    weight = 0.2  # default
                weighted_contrib_data.append({
                    'Ticker': item['Ticker'],
                    'Factor': factor.replace('_', ' '),
                    'Weighted Contribution': score * weight
                })
        
        weighted_contrib_df = pd.DataFrame(weighted_contrib_data)
        
        # Create a stacked bar chart for weighted contribution
        fig2 = px.bar(
            weighted_contrib_df,
            x='Ticker',
            y='Weighted Contribution',
            color='Factor',
            title="Weighted Contribution to Overall Score",
            color_discrete_sequence=px.colors.qualitative.Plotly
        )
        
        fig2.update_layout(
            xaxis_title="Ticker",
            yaxis_title="Weighted Contribution",
            legend_title="Factor",
            height=500
        )
        
        st.plotly_chart(fig2, use_container_width=True)
        
        # Overall Score Comparison
        st.markdown("### 🏆 Overall Score Comparison")
        
        overall_scores_df = pd.DataFrame([
            {
                'Ticker': item['Ticker'],
                'Overall Score': item['overall_score'],
                'Recommendation': item['recommendation']
            }
            for item in detailed_scores_list
        ])
        
        # Create a bar chart for overall scores
        fig3 = px.bar(
            overall_scores_df,
            x='Ticker',
            y='Overall Score',
            color='Recommendation',
            title="Overall Score Comparison",
            color_discrete_map={
                'Strong Buy': '#2e7d32',
                'Buy': '#8bc34a',
                'Hold': '#ffeb3b',
                'Sell': '#f44336'
            }
        )
        
        fig3.update_layout(
            xaxis_title="Ticker",
            yaxis_title="Overall Score (1-5)",
            height=500
        )
        
        st.plotly_chart(fig3, use_container_width=True)
        
        # Factor Performance Heatmap
        st.markdown("### 🔥 Factor Performance Heatmap")
        
        # Prepare data for heatmap
        heatmap_data = []
        for item in detailed_scores_list:
            heatmap_row = {'Ticker': item['Ticker']}
            for factor, data in item['factor_scores'].items():
                score = data['score']
                if isinstance(score, (list, dict, tuple)):
                    score = 3.0  # default
                heatmap_row[factor.replace('_', ' ')] = score
            heatmap_data.append(heatmap_row)
        
        heatmap_df = pd.DataFrame(heatmap_data)
        heatmap_df.set_index('Ticker', inplace=True)
        
        # Create heatmap
        fig4 = px.imshow(
            heatmap_df,
            labels=dict(x="Factor", y="Ticker", color="Score"),
            color_continuous_scale='RdYlGn',
            title="Factor Performance Heatmap",
            aspect="auto"
        )
        
        fig4.update_layout(height=500)
        
        st.plotly_chart(fig4, use_container_width=True)
        
        # Factor Distribution
        st.markdown("### 📊 Factor Score Distribution")
        
        # Prepare data for distribution
        distribution_data = []
        for item in detailed_scores_list:
            for factor, data in item['factor_scores'].items():
                score = data['score']
                if isinstance(score, (list, dict, tuple)):
                    score = 3.0  # default
                distribution_data.append({
                    'Factor': factor.replace('_', ' '),
                    'Score': score
                })
        
        distribution_df = pd.DataFrame(distribution_data)
        
        # Create violin plot
        fig5 = px.violin(
            distribution_df,
            x='Factor',
            y='Score',
            box=True,
            points="all",
            title="Factor Score Distribution",
            color_discrete_sequence=px.colors.qualitative.Plotly
        )
        
        fig5.update_layout(
            xaxis_title="Factor",
            yaxis_title="Score (1-5)",
            height=500,
            showlegend=False
        )
        
        st.plotly_chart(fig5, use_container_width=True)
    
    with tab7:  # Advanced Analytics tab
        st.markdown("## 🧠 Advanced Analytics")
        
        # Score Distribution
        st.markdown("### 📊 Score Distribution")
        fig = plot_score_distribution(detailed_scores_list)
        st.plotly_chart(fig, use_container_width=True)
        
        # Factor Contributions
        st.markdown("### ⚖️ Factor Contributions")
        fig = plot_factor_contributions(detailed_scores_list)
        st.plotly_chart(fig, use_container_width=True)
        
        # PCA Analysis
        st.markdown("### 🧩 PCA Analysis")
        fig = plot_pca_analysis(fundamentals_df)
        if fig:
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.warning("Insufficient data to perform PCA analysis")
        
        # Correlation Matrix
        st.markdown("### 🔗 Correlation Matrix")
        
        # Select numeric columns
        numeric_cols = []
        for col in fundamentals_df.columns:
            try:
                # Try to convert to numeric
                pd.to_numeric(fundamentals_df[col], errors='raise')
                numeric_cols.append(col)
            except:
                pass
        
        if len(numeric_cols) >= 3:
            # Prepare data
            df = fundamentals_df[numeric_cols].copy()
            
            # Convert to numeric
            for col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce')
            
            # Drop rows with NaN
            df = df.dropna()
            
            if len(df) >= 2:
                # Calculate correlation matrix
                corr_matrix = df.corr()
                
                # Create heatmap
                fig = px.imshow(
                    corr_matrix,
                    labels=dict(x="Metrics", y="Metrics", color="Correlation"),
                    color_continuous_scale='RdBu',
                    zmid=0,
                    title="Financial Metrics Correlation Matrix"
                )
                
                fig.update_layout(height=600)
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.warning("Insufficient data to plot correlation matrix")
        else:
            st.warning("Insufficient numeric data to plot correlation matrix")
    
    with st.expander("⚙️ Raw data (for debugging)"):
        st.write("Fundamentals Data:")
        st.write(fundamentals_df)
        st.write("Detailed Scores Data:")
        st.write(pd.DataFrame([
            {
                'Ticker': item['Ticker'],
                'Overall Score': item['overall_score'],
                'Recommendation': item['recommendation'],
                'Management & Governance': item['factor_scores']['Management_Governance']['score'],
                'Earnings Quality': item['factor_scores']['Earnings_Quality']['score'],
                'Profitability': item['factor_scores']['Profitability']['score'],
                'Valuation': item['factor_scores']['Valuation']['score'],
                'Balance Sheet': item['factor_scores']['Balance_Sheet']['score'],
                'Cash Flow': item['factor_scores']['Cash_Flow']['score']
            }
            for item in detailed_scores_list
        ]))