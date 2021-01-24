import yfinance as yf
import streamlit as st
import pandas as pd

st.write("""
# Simple Stock Price App

Shown are the Stock Closing Price and Volume of Google!

""")

tickersymbol = 'GOOGL'

tickerdata = yf.Ticker(tickersymbol)

tickerDf = tickerdata.history(period='Id',start='2010-5-31', end='2020-5-31')

st.line_chart(tickerDf.Close)
st.line_chart(tickerDf.Volume)