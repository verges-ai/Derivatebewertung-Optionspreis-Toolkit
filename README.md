# 📊 Optionsanalyse, Greeks & KI-Modul

Dieses Projekt kombiniert klassische Finanzmathematik (Black-Scholes-Modell) mit modernen Data-Science- und Machine-Learning-Methoden. Es bietet eine interaktive Streamlit-Webanwendung zur Berechnung von Optionspreisen, Greeks und zur Prognose von Volatilität mithilfe künstlicher Intelligenz.

---

## 🚀 Features

- ✅ **Berechnung von Call- und Put-Preisen** nach dem Black-Scholes-Modell
- ✅ **Greeks-Analyse** (Delta, Gamma, Vega, Theta, Rho)
- ✅ **Interaktive Visualisierung**: Greeks vs. Volatilität, Laufzeit, Kurs
- ✅ **KI-Modul** zur Prognose künftiger Volatilitäten (Random Forest)
- ✅ **CSV-Upload** für Preiszeitreihen (historische Daten)
- ✅ **Klares UI** dank Streamlit

---

## 🧠 Verwendete Technologien

- Python 3.10+
- NumPy, Pandas
- Scikit-Learn
- Matplotlib
- Streamlit

---

## 📂 Projektstruktur

```bash
.
├── app.py                          # Haupt-Streamlit-Anwendung
├── pricing/
│   ├── __init__.py
│   └── black_scholes.py           # Preis- und Greeks-Funktionen
├── data/
│   └── prices.csv                 # Beispiel-Daten für KI
├── README.md
└── requirements.txt

---

## 🌍 Live Demo

👉 [Click here to open the app](https://your-app-name.streamlit.app)






