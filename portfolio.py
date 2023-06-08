import pandas as pd
import yfinance as yf
import matplotlib.pyplot as plt
import numpy as np
import streamlit as st
from datetime import datetime, timedelta

st.title('Investment Portfolio Dashboard')

#Provide defult assets and start time

assets = st.text_input("Provide at least two assets (comma-separated)","AAPL,MSFT,GOOGL")
default_date = datetime.now() - timedelta(days=365)
start = st.date_input("Pick a starting date for your analysis", value=pd.to_datetime(default_date))

#pull data and get a PRICE dataframe with only adj close 
data = yf.download(assets,start=start)['Adj Close']

#calculate daily % returns instead of prices and then cummulating them
ret_df = data.pct_change()
cumul_ret= (ret_df+1).cumprod()-1

#getting a simple average portfolio return given that each asset is weighted equally
pf_cumul_ret = cumul_ret.mean(axis=1)

#same thing as above: get data for the index, cummulate it as a %
benchmark = yf.download('^GSPC',start=start)['Adj Close']
bench_ret = benchmark.pct_change()
bench_dev = (bench_ret+1).cumprod()-1


#next: Portfolio rist calculation

#Create numpy array of ones, three times
#Divide by three (i.e. length) to get equal weights
W = (np.ones(len(ret_df.cov()))/len(ret_df.cov()))

#this one is tricky, but essentially get the covarience, weight it and then sqrt for stdev
pf_std = (W.dot(ret_df.cov()).dot(W) **(1/2))


#next: Plotting part
st.subheader('Portfolio vs. Index (% return)')

tog = pd.concat([bench_dev*100, pf_cumul_ret*100],axis=1)
tog.columns = ['S&P500 Performance', 'Portfolio Performance']


st.line_chart(data=tog)

st.subheader("Portfolio Risk (σ):")
pf_std*100
st.subheader("Benchmark Risk (σ):")
bench_risk = bench_ret.std()
bench_risk*100

if pf_std > bench_risk:
    st.markdown('<p style="color:red;">Portfolio is risky! Consider adding more assets.</p>', unsafe_allow_html=True)
else:
    st.markdown('<p style="color:green;">Portfolio risk is reduced!</p>', unsafe_allow_html=True)


st.subheader('Portfolio composition:')

fig, ax = plt.subplots(figsize=(6, 6), facecolor='white')
ax.pie(W, labels=data.columns, autopct='%.1f%%', textprops={'color': 'black'})


st.pyplot(fig)
