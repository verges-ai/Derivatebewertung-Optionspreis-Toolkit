# app.py
import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

# Import aus eigenem Modul
from pricing.black_scholes import black_scholes_price, greeks

# 📌 Titel
st.title("📊 Optionsanalyse, Greeks & KI-Modul")

# --- 📥 Eingaben ---
st.sidebar.header("Eingabewerte")
S = st.sidebar.number_input("Aktueller Kurs (S)", value=100.0)
K = st.sidebar.number_input("Strike-Preis (K)", value=100.0)
T = st.sidebar.number_input("Restlaufzeit (in Jahren)", value=1.0)
r = st.sidebar.number_input("Risikofreier Zinssatz (r)", value=0.02)
sigma = st.sidebar.slider("Volatilität (σ)", 0.05, 1.0, 0.25, 0.01)
option_type = st.sidebar.selectbox("Optionstyp", ["call", "put"])

# --- 💰 Preis & Greeks ---
st.header("💰 Preis & Greeks")
price = black_scholes_price(S, K, T, r, sigma, option_type)
greek_vals = greeks(S, K, T, r, sigma)[option_type]
st.markdown(f"**{option_type.capitalize()}-Preis**: `{price:.2f}` EUR")
st.write(greek_vals)

# --- 📊 Visualisierung ---
st.header("📈 Greeks Visualisierung")
selected_greek = st.selectbox("Greek wählen:", list(greek_vals.keys()))
x_axis = st.selectbox("X-Achse:", ["Volatilität (σ)", "Restlaufzeit (T)", "Aktienkurs (S)"])

if x_axis == "Volatilität (σ)":
    x_vals = np.linspace(0.05, 1.0, 50)
    compute = lambda x: greeks(S, K, T, r, x)[option_type][selected_greek]
    xlabel = "Volatilität (σ)"
elif x_axis == "Restlaufzeit (T)":
    x_vals = np.linspace(0.01, 2.0, 50)
    compute = lambda x: greeks(S, K, x, r, sigma)[option_type][selected_greek]
    xlabel = "Restlaufzeit (T)"
else:
    x_vals = np.linspace(50, 150, 50)
    compute = lambda x: greeks(x, K, T, r, sigma)[option_type][selected_greek]
    xlabel = "Aktienkurs (S)"

y_vals = [compute(x) for x in x_vals]
fig, ax = plt.subplots()
ax.plot(x_vals, y_vals, label=selected_greek, color="teal")
ax.set_xlabel(xlabel)
ax.set_ylabel(selected_greek)
ax.set_title(f"{selected_greek} vs. {xlabel} ({option_type})")
ax.legend()
ax.grid(True)
st.pyplot(fig)

# --- 🔮 KI-Modul: Volatilitätsprognose ---
st.header("🔮 KI-Modul: Volatilitätsprognose")
uploaded_file = st.file_uploader("📂 Lade Preis-Daten (CSV, Spalten: Date, Close)", type="csv")

if uploaded_file is not None:
    df = pd.read_csv(uploaded_file, parse_dates=["Date"])
    df.sort_values("Date", inplace=True)
    df["log_return"] = np.log(df["Close"] / df["Close"].shift(1))
    df["volatility"] = df["log_return"].rolling(window=20).std() * np.sqrt(252)
    df["prev_vol"] = df["volatility"].shift(1)
    df.dropna(inplace=True)

    X = df[["prev_vol"]]
    y = df["volatility"]
    X_train, X_test, y_train, y_test = train_test_split(X, y, shuffle=False, test_size=0.2)

    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)

    st.subheader("📈 Modellgüte")
    st.write(f"MSE: `{mse:.6f}`")

    last_vol = X.iloc[-1].values.reshape(1, -1)
    predicted_vol = model.predict(last_vol)[0]
    st.subheader("🔮 Vorhergesagte nächste Volatilität")
    st.write(f"**{predicted_vol:.4f}**")

    fig2, ax2 = plt.subplots()
    ax2.plot(df["Date"].iloc[-len(y_test):], y_test, label="Tatsächlich", color="blue")
    ax2.plot(df["Date"].iloc[-len(y_test):], y_pred, label="Prognose", color="red")
    ax2.set_title("Volatilitätsprognose")
    ax2.legend()
    st.pyplot(fig2)

st.caption("🧠 Erstellt mit Liebe zum quantitativen Handwerk • Projekt 1 von 3")
