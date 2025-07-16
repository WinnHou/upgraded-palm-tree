# trump_predictor/app.py

import streamlit as st
import pandas as pd
import requests
import joblib
import os
import numpy as np
from datetime import datetime, timedelta
import torch
import torch.nn as nn
from sklearn.preprocessing import MinMaxScaler
from scipy.signal import savgol_filter
from streamlit_autorefresh import st_autorefresh
from sklearn.metrics import mean_absolute_error, mean_squared_error

# 尝试导入 Plotly
try:
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots
except ImportError:
    st.error("缺少依赖：plotly。请运行 `pip install plotly` 后重试。")
    st.stop()

# 自动刷新
st_autorefresh(interval=60 * 1000, key="data_refresh")

# 参数配置
SYMBOL = "TRUMPUSDT"
INTERVAL = "15m"
LIMIT = 500
SEQ_LEN = 10
INPUT_SIZE = 5  # open, high, low, close, volume
HIDDEN_SIZE = 64
FUTURE_STEPS = 4  # 预测未来 1 小时
MODEL_PATH = "lstm_model.pt"
SCALER_PATH = "scaler.pkl"
PERIOD = 50  # 可视化最近数据点数量

# Streamlit 页面设置
st.set_page_config(page_title="Trump 币价格预测", layout="wide")
st.title("Trump 币 15 分钟价格预测")

# 全局样式
st.markdown("""
<style>
    .stMetricValue { font-size: 22px; }
    .stCaption { font-size: 14px; color: gray; }
</style>
""", unsafe_allow_html=True)

# 获取 K 线数据
@st.cache_data(ttl=60)
def fetch_kline(symbol, interval, limit):
    url = "https://api.binance.com/api/v3/klines"
    params = {"symbol": symbol, "interval": interval, "limit": limit}
    data = requests.get(url, params=params).json()
    df = pd.DataFrame(data, columns=[
        "open_time","open","high","low","close","volume",
        "close_time","quote_asset_volume","num_trades","taker_buy_base",
        "taker_buy_quote","ignore"
    ])
    df[["open","high","low","close","volume"]] = df[["open","high","low","close","volume"]].astype(float)
    df["close_time"] = pd.to_datetime(df["close_time"], unit="ms")
    return df[["close_time","open","high","low","close","volume"]]

# 构造样本序列
def build_sequences(data, seq_len):
    X, y, times = [], [], []
    for i in range(len(data) - seq_len):
        X.append(data[i: i+seq_len])
        y.append(data[i+seq_len, 3])  # close 是索引 3
        times.append(i + seq_len)
    return np.array(X), np.array(y), times

# 定义 LSTM 模型
def create_model(input_size, hidden_size):
    class LSTMModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.lstm = nn.LSTM(input_size, hidden_size, num_layers=2,
                                 batch_first=True, dropout=0.2)
            self.fc = nn.Linear(hidden_size, 1)
        def forward(self, x):
            out, _ = self.lstm(x)
            return self.fc(out[:, -1, :]).squeeze()
    return LSTMModel()

# 训练模型
@st.cache_data()
def train_model(df):
    data = df[["open","high","low","close","volume"]].values
    scaler = MinMaxScaler()
    data_s = scaler.fit_transform(data)
    X, y, _ = build_sequences(data_s, SEQ_LEN)
    X_t = torch.tensor(X, dtype=torch.float32)
    y_t = torch.tensor(y, dtype=torch.float32)
    model = create_model(INPUT_SIZE, HIDDEN_SIZE)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.MSELoss()
    for epoch in range(200):
        model.train()
        optimizer.zero_grad()
        out = model(X_t)
        loss = criterion(out, y_t)
        loss.backward()
        optimizer.step()
    torch.save(model.state_dict(), MODEL_PATH)
    joblib.dump(scaler, SCALER_PATH)
    return model, scaler

# 预测并评估
def predict(df, model, scaler):
    data = df[["open","high","low","close","volume"]].values
    data_s = scaler.transform(data)
    X, y_true, times = build_sequences(data_s, SEQ_LEN)
    X_t = torch.tensor(X, dtype=torch.float32)
    model.eval()
    with torch.no_grad():
        y_pred_s = model(X_t).numpy()
    close_min = scaler.data_min_[3]
    close_range = scaler.data_max_[3] - scaler.data_min_[3]
    y_pred = y_pred_s * close_range + close_min
    y_true_vals = df["close"].values[SEQ_LEN:]
    mae = mean_absolute_error(y_true_vals, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true_vals, y_pred))
    future_s, future_times = [], []
    last_seq = data_s[-SEQ_LEN:].copy()
    for i in range(FUTURE_STEPS):
        inp = torch.tensor(last_seq[np.newaxis], dtype=torch.float32)
        with torch.no_grad():
            fp_s = model(inp).item()
        future_s.append(fp_s)
        new_line = last_seq[-1].copy()
        new_line[3] = fp_s
        last_seq = np.vstack([last_seq[1:], new_line])
        future_times.append(df["close_time"].iloc[-1] + timedelta(minutes=15*(i+1)))
    future_pred = np.array(future_s) * close_range + close_min
    pred_times = [df["close_time"].iloc[i] for i in times]
    return y_pred, y_true_vals, pred_times, future_pred, future_times, mae, rmse

# 绘制交互式K线与预测
def plot_candlestick(df, pred_times, y_pred, future_times, future_pred):
    df_plot = df.iloc[-PERIOD:].copy()
    if len(pred_times) >= PERIOD:
        pt = pred_times[-PERIOD:]
        yp = y_pred[-PERIOD:]
    else:
        pt, yp = pred_times, y_pred
    if len(yp) >= 9:
        yp = savgol_filter(yp, 9, 2)
    fig = make_subplots(rows=1, cols=1, shared_xaxes=True)
    fig.add_trace(go.Candlestick(
        x=df_plot["close_time"], open=df_plot["open"],
        high=df_plot["high"], low=df_plot["low"], close=df_plot["close"],
        name="K线"), row=1, col=1)
    fig.add_trace(go.Scatter(
        x=pt, y=yp, mode="lines", name="历史预测",
        line=dict(color="#ff7f0e", dash="dash")), row=1, col=1)
    fig.add_trace(go.Scatter(
        x=future_times, y=future_pred, mode="lines+markers", name="未来预测",
        line=dict(color="green", dash="dot")), row=1, col=1)
    fig.update_layout(
        xaxis_rangeslider_visible=False,
        margin=dict(l=40, r=40, t=40, b=40),
        template="plotly_dark"
    )
    fig.update_yaxes(tickformat=".3f")
    fig.update_xaxes(type="date", tickformat="%H:%M", tickangle=-45)
    st.plotly_chart(fig, use_container_width=True)

# 主程序

df = fetch_kline(SYMBOL, INTERVAL, LIMIT)
if df.empty:
    st.error("获取数据失败，停止程序。")
    st.stop()

if os.path.exists(MODEL_PATH) and os.path.exists(SCALER_PATH):
    model = create_model(INPUT_SIZE, HIDDEN_SIZE)
    try:
        model.load_state_dict(torch.load(MODEL_PATH))
        scaler = joblib.load(SCALER_PATH)
        st.info("模型已加载。")
    except:
        st.warning("模型加载失败，正在训练...")
        model, scaler = train_model(df)
        st.success("训练完成并保存。")
else:
    model, scaler = train_model(df)
    st.success("训练完成并保存。")

# 预测
y_pred, y_true, pred_times, future_pred, future_times, mae, rmse = predict(df, model, scaler)

# 指标展示
col1, col2 = st.columns(2)
col1.metric("MAE", f"{mae:.4f}")
col2.metric("RMSE", f"{rmse:.4f}")

# 下一预测
st.subheader("下一个收盘价预测")
st.metric("预测值", f"{future_pred[0]:.3f} USDT")

# 图表
st.subheader("实际 K 线 与 预测 曲线")
plot_candlestick(df, pred_times, y_pred, future_times, future_pred)

# 更新时间
st.caption(f"更新时间：{datetime.now().strftime('%Y-%m-%d %H:%M:%S')} | 数据来源：Binance")
