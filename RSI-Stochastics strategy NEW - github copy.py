# -*- coding: utf-8 -*-
"""
Created on Thu Feb  4 09:45:22 2021

@author: Amogh
"""

# standard imports
import warnings
warnings.filterwarnings('ignore')

import pandas as pd
import numpy as np
import datetime as dt
import yfinance as yf
import matplotlib.pyplot as plt
import pyfolio as pf


from IPython.core.interactiveshell import InteractiveShell
InteractiveShell.ast_node_interactivity = "all"

pd.set_option('display.max_rows', None)


# Fetching data
stocks = ["ADANIPORTS.NS","ASIANPAINT.NS","AXISBANK.NS","BAJAJ-AUTO.NS","BAJAJFINSV.NS",
          "BAJFINANCE.NS","BHARTIARTL.NS","BPCL.NS","BRITANNIA.NS","CIPLA.NS",
          "COALINDIA.NS","DRREDDY.NS","EICHERMOT.NS","GAIL.NS","GRASIM.NS",
          "HCLTECH.NS","HDFC.NS","HDFCBANK.NS","HEROMOTOCO.NS","HINDALCO.NS",
          "HINDUNILVR.NS","ICICIBANK.NS","INDUSINDBK.NS","INFY.NS",
          "IOC.NS","ITC.NS","JSWSTEEL.NS","KOTAKBANK.NS","LT.NS",
          "M&M.NS","MARUTI.NS","NESTLEIND.NS","NTPC.NS","ONGC.NS",
          "POWERGRID.NS","RELIANCE.NS","SBIN.NS","SHREECEM.NS","SUNPHARMA.NS",
          "TATAMOTORS.NS","TATASTEEL.NS","TCS.NS","TECHM.NS","TITAN.NS",
          "ULTRACEMCO.NS","UPL.NS","VEDL.NS","WIPRO.NS","ZEEL.NS"]


start = dt.datetime.today()-dt.timedelta(days=2520)
end = dt.datetime.today()
ohlcv_data = {}

for ticker in stocks:
    ohlcv_data[ticker] = yf.download(ticker,start,end).dropna(how="all")

# Converting to Adjusted Prices(For Open, High, Low, Close)
for ticker in stocks:
    ohlcv_data[ticker]["Open"] =  (ohlcv_data[ticker]["Adj Close"]/ ohlcv_data[ticker]["Close"])*ohlcv_data[ticker]["Open"] 
    ohlcv_data[ticker]["High"] =  (ohlcv_data[ticker]["Adj Close"]/ ohlcv_data[ticker]["Close"])*ohlcv_data[ticker]["High"]
    ohlcv_data[ticker]["Low"] =  (ohlcv_data[ticker]["Adj Close"]/ ohlcv_data[ticker]["Close"])*ohlcv_data[ticker]["Low"]
    ohlcv_data[ticker]["Close"] =  (ohlcv_data[ticker]["Adj Close"]/ ohlcv_data[ticker]["Close"])*ohlcv_data[ticker]["Close"]


# Indicators and SL level computation:
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
    #df['ATR'] = df['TR'].ewm(span=n,adjust=False,min_periods=n).mean()
    #df['roll_max_cp'] = df["High"].rolling(n).max()
    #df['roll_min_cp'] = df["Low"].rolling(n).min()
    # Made changes here
    df['ch_long'] = df['High'].rolling(n).max() - df['ATR']*(m)
    df['ch_short'] = df['Low'].rolling(n).min() + df['ATR']*(m)
    return df['ch_long'],df['ch_short']



# Calculating Indicators and SL levels and dropping NaNs
ohlc_dict = ohlcv_data.copy()
for ticker in stocks:

    ohlc_dict[ticker]["Stochastics"] = STOCH(ohlc_dict[ticker],20)
    ohlc_dict[ticker]["RSI"] = RSI(ohlc_dict[ticker],250)
    ohlc_dict[ticker]["Ch_long"] = CHSTOP(ohlc_dict[ticker],22,2)[0]
    ohlc_dict[ticker]["Ch_sh"] = CHSTOP(ohlc_dict[ticker],22,2)[1]

    ohlc_dict[ticker].dropna(inplace=True)
    #print(ohlcv_data[ticker].info())
    

# SIGNAL GENERATION
# Generating signals/positions
for ticker in stocks:
    df = ohlc_dict[ticker]
    signal=[0]
    position = "flat"
    
    for i in range(1,len(df)):
        
        if position == "flat":
            if (df["RSI"].iloc[i] >= 50) and (df["Stochastics"].iloc[i]>25) and (df["Stochastics"].iloc[i-1]<20):
                position = "long"
                signal.append("long")
            elif (df["RSI"].iloc[i] < 50) and (df["Stochastics"].iloc[i]<75) and (df["Stochastics"].iloc[i-1]>80):
                position = "short"
                signal.append("short")
            else:
                position = "flat"
                signal.append("flat")
                
                
        elif position =="long":
            if df["Adj Close"].iloc[i]< df["Ch_long"].iloc[i-1]: 
                position = "flat"
                signal.append("flat")
            else:
                position = "long"
                signal.append("long")
                
                
        elif position =="short":
            if df["Adj Close"].iloc[i]> df["Ch_sh"].iloc[i-1]: 
                position = "flat"
                signal.append("flat")
            else:
                position = "short"
                signal.append("short")
                
        #print(ticker, position)
    #print(len(df))
    #print(len(signal))
    ohlc_dict[ticker]['position'] = signal
    
    
    
# Calculating returns
for ticker in stocks:
    df = ohlc_dict[ticker]
    
    df['ret'] = 0
    
    for i in range(1,len(df)):

        if df['position'].iloc[i-1] == "long":
            if df['position'].iloc[i] == df['position'].iloc[i-1]:
                df['ret'].iloc[i] = (df['Adj Close'].iloc[i]/df['Adj Close'].iloc[i-1])-1

            if df['position'].iloc[i] != df['position'].iloc[i-1]:
                df['ret'].iloc[i] = (df['Ch_long'].iloc[i-1]/df['Adj Close'].iloc[i-1])-1

        elif df['position'].iloc[i-1] == "short":
            if df['position'].iloc[i] == df['position'].iloc[i-1]:
                df['ret'].iloc[i] = -1*((df['Adj Close'].iloc[i]/df['Adj Close'].iloc[i-1])-1)

            if df['position'].iloc[i] != df['position'].iloc[i-1]:
                df['ret'].iloc[i] = (df['Adj Close'].iloc[i-1]/df['Ch_sh'].iloc[i-1])-1
    
    
# USING PYFOLIO
#for ticker in stocks:
    #print(f'PERFORMANCE FOR {ticker}')
    #pf.create_simple_tear_sheet(ohlc_dict[ticker]["ret"])    
    
    

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
    




#calculating individual stock's KPIs
cagr = {}
sharpe_ratios = {}
for ticker in stocks:
    print("calculating KPIs for ",ticker)      
    cagr[ticker] =  CAGR(ohlc_dict[ticker])
    sharpe_ratios[ticker] =  sharpe(ohlc_dict[ticker],0.067)          

KPI_df = pd.DataFrame([cagr,sharpe_ratios],index=["CAGR","Sharpe Ratio"]).T   
