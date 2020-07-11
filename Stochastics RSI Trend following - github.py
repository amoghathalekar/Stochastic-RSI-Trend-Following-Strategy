# -*- coding: utf-8 -*-
"""
Created on Tue May 26 12:10:27 2020

@author: Amogh
"""

import datetime as dt
import yfinance as yf
import pandas as pd
import numpy as np

stocks = ["ADANIPORTS.NS","ASIANPAINT.NS","AXISBANK.NS","BAJAJ-AUTO.NS","BAJAJFINSV.NS",
          "BAJFINANCE.NS","BHARTIARTL.NS","BPCL.NS","BRITANNIA.NS","CIPLA.NS",
          "COALINDIA.NS","DRREDDY.NS","EICHERMOT.NS","GAIL.NS","GRASIM.NS",
          "HCLTECH.NS","HDFC.NS","HDFCBANK.NS","HEROMOTOCO.NS","HINDALCO.NS",
          "HINDUNILVR.NS","ICICIBANK.NS","INDUSINDBK.NS","INFRATEL.NS","INFY.NS",
          "IOC.NS","ITC.NS","JSWSTEEL.NS","KOTAKBANK.NS","LT.NS",
          "M&M.NS","MARUTI.NS","NESTLEIND.NS","NTPC.NS","ONGC.NS",
          "POWERGRID.NS","RELIANCE.NS","SBIN.NS","SHREECEM.NS","SUNPHARMA.NS",
          "TATAMOTORS.NS","TATASTEEL.NS","TCS.NS","TECHM.NS","TITAN.NS",
          "ULTRACEMCO.NS","UPL.NS","VEDL.NS","WIPRO.NS","ZEEL.NS"]

start = dt.datetime.today()-dt.timedelta(5900)
end = dt.datetime.today()
ohlcv_data = {}


for ticker in stocks:
    ohlcv_data[ticker] = yf.download(ticker,start,end).dropna(how="all")
    

#XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX

#DEFINING KPIs:
def CAGR(DF):
    "function to calculate the Cumulative Annual Growth Rate of a trading strategy"
    df = DF.copy()
    df["cum_return"] = (1 + df["ret"]).cumprod()
    n = len(df)/(252)
    CAGR = (df["cum_return"].tolist()[-1])**(1/n) - 1
    return CAGR

def volatility(DF):
    "function to calculate annualized volatility of a trading strategy"
    df = DF.copy()
    vol = df["ret"].std() * np.sqrt(252)
    return vol

def sharpe(DF,rf):
    "function to calculate sharpe ratio ; rf is the risk free rate"
    df = DF.copy()
    sr = (CAGR(df) - rf)/volatility(df)
    return sr
    

def max_dd(DF):
    "function to calculate max drawdown"
    df = DF.copy()
    df["cum_return"] = (1 + df["ret"]).cumprod()
    df["cum_roll_max"] = df["cum_return"].cummax()
    df["drawdown"] = df["cum_roll_max"] - df["cum_return"]
    df["drawdown_pct"] = df["drawdown"]/df["cum_roll_max"]
    max_dd = df["drawdown_pct"].max()
    return max_dd



#XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX

    
#DEFINING RSI:    
def RSI(DF,n):
    "function to calculate RSI"
    df = DF.copy()
    df['delta']=df['Close'] - df['Close'].shift(1)
    df['gain']=np.where(df['delta']>=0,df['delta'],0)
    df['loss']=np.where(df['delta']<0,abs(df['delta']),0)
    avg_gain = []
    avg_loss = []
    gain = df['gain'].tolist()
    loss = df['loss'].tolist()
    for i in range(len(df)):
        if i < n:
            avg_gain.append(np.NaN)
            avg_loss.append(np.NaN)
        elif i == n:
            avg_gain.append(df['gain'].rolling(n).mean().tolist()[n])
            avg_loss.append(df['loss'].rolling(n).mean().tolist()[n])
        elif i > n:
            avg_gain.append(((n-1)*avg_gain[i-1] + gain[i])/n)
            avg_loss.append(((n-1)*avg_loss[i-1] + loss[i])/n)
    df['avg_gain']=np.array(avg_gain)
    df['avg_loss']=np.array(avg_loss)
    df['RS'] = df['avg_gain']/df['avg_loss']
    df['RSI'] = 100 - (100/(1+df['RS']))
    return df['RSI']



#DEFINING STOCHASTICS:
# %K = (Current close - lowest low)/(Highest High - Lowest Low) * 100
# %d = 3-day SMA of %K ...not needed with our strategy
# Lowest low = Lowest low for the lookback period
# Highest High = Highest High for the lookback period

def STOCH(DF,n):
    "function to calculate Stochastics"
    df = DF.copy()
    df['cc-ll'] = df['Close'] - df['Low'].rolling(n).min()
    df['hh-ll'] = df['High'].rolling(n).max() - df['Low'].rolling(n).min()
    df['stoch'] = df['cc-ll']/df['hh-ll']*100
    return df['stoch']


#DEFINING ATR:(already coded in Chandelier Stop)
#def ATR(DF,n):
#    "function to calculate True Range and Average True Range"
#   df = DF.copy()
#   df['H-L']=abs(df['High']-df['Low'])
#    df['H-PC']=abs(df['High']-df['Close'].shift(1))
#    df['L-PC']=abs(df['Low']-df['Close'].shift(1))
#    df['TR']=df[['H-L','H-PC','L-PC']].max(axis=1,skipna=False)
#    df['ATR'] = df['TR'].rolling(n).mean()
#    #df['ATR'] = df['TR'].ewm(span=n,adjust=False,min_periods=n).mean()
#    df2 = df.drop(['H-L','H-PC','L-PC'],axis=1)
#    return df2


#DEFINING CHANDELIER STOP:
def CHSTOP(DF,n,m):
    "function for estimation of Chandelier Stop Loss/Exit"
    df = DF.copy()
    df['H-L']=abs(df['High']-df['Low'])
    df['H-PC']=abs(df['High']-df['Close'].shift(1))
    df['L-PC']=abs(df['Low']-df['Close'].shift(1))
    df['TR']=df[['H-L','H-PC','L-PC']].max(axis=1,skipna=False)
    df['ATR'] = df['TR'].rolling(n).mean()
    df['roll_max_cp'] = df["High"].rolling(n).max()
    df['roll_min_cp'] = df["Low"].rolling(n).min()
    df['ch_long'] = df['High'].rolling(n).max() - df['ATR']*(m)
    df['ch_short'] = df['Low'].rolling(n).min() + df['ATR']*(m)
    return df['ch_long'],df['ch_short']
    

#####################################   BACKTESTING   ##################################
#Working on the ACTUAL STRATEGY:
ohlc_dict = ohlcv_data.copy()
tickers_signal = {}
tickers_ret = {}
for ticker in stocks:
    print("Calculating RSI, Stochastics, Chandelier Stops for ",ticker)
    ohlc_dict[ticker]["Stochastics"] = STOCH(ohlc_dict[ticker],20)
    ohlc_dict[ticker]["RSI"] = RSI(ohlc_dict[ticker],250)
    ohlc_dict[ticker]["Ch_long"] = CHSTOP(ohlc_dict[ticker],22,2)[0]
    ohlc_dict[ticker]["Ch_sh"] = CHSTOP(ohlc_dict[ticker],22,2)[1]
    tickers_signal[ticker] = ""
    tickers_ret[ticker] = []
    
for ticker in stocks:
    ohlc_dict[ticker].dropna(inplace=True)
    
#Identifying Signals and calculating daily return (Stop Loss factored in)
for ticker in stocks:
    print("Calculating returns for ",ticker)
    for i in range(len(ohlc_dict[ticker])):
        if tickers_signal[ticker] == "":
            tickers_ret[ticker].append(0)
            if ohlc_dict[ticker]["RSI"][i]>=50 and \
               ohlc_dict[ticker]["Stochastics"][i]>25 and ohlc_dict[ticker]["Stochastics"][i-1]<20:
                   tickers_signal[ticker] = "Buy"
            elif ohlc_dict[ticker]["RSI"][i]<50 and \
                 ohlc_dict[ticker]["Stochastics"][i]<75 and ohlc_dict[ticker]["Stochastics"][i-1]>80:
                    tickers_signal[ticker] = "Sell"
                     
        elif tickers_signal[ticker] == "Buy":
            if ohlc_dict[ticker]["Close"][i]<ohlc_dict[ticker]["Ch_long"][i-1]: #change made:["CH_long"][i-1]
                tickers_signal[ticker] = ""
                tickers_ret[ticker].append((ohlc_dict[ticker]["Ch_long"][i-1]/ohlc_dict[ticker]["Close"][i-1])-1) #change made:["CH_long"][i-1]
            else:
                tickers_ret[ticker].append((ohlc_dict[ticker]["Close"][i]/ohlc_dict[ticker]["Close"][i-1])-1)
                
                
        elif tickers_signal[ticker] == "Sell":
            if ohlc_dict[ticker]["Close"][i]>ohlc_dict[ticker]["Ch_sh"][i-1]: #change made:["CH_sh"][i-1]
               tickers_signal[ticker] = ""
               tickers_ret[ticker].append((ohlc_dict[ticker]["Close"][i-1]/ohlc_dict[ticker]["Ch_sh"][i-1])-1) #change made:["CH_sh"][i-1]
            else:
               tickers_ret[ticker].append((ohlc_dict[ticker]["Close"][i-1]/ohlc_dict[ticker]["Close"][i])-1)
                
                
    ohlc_dict[ticker]["ret"] = np.array(tickers_ret[ticker])
                        
                     
#Calculating overall strategy's KPIs:
strategy_df = pd.DataFrame()
for ticker in stocks:
    strategy_df[ticker] = ohlc_dict[ticker]["ret"]
strategy_df["ret"] = strategy_df.mean(axis=1)
      
                     
#Calculating individual stock's KPIs
cagr = {}
sharpe_ratios = {}
for ticker in stocks:
    print("calculating KPIs for ",ticker)      
    cagr[ticker] =  CAGR(ohlc_dict[ticker])
    sharpe_ratios[ticker] =  sharpe(ohlc_dict[ticker],0.067)          

KPI_df = pd.DataFrame([cagr,sharpe_ratios],index=["Return","Sharpe Ratio"]).T   
KPI_df

           