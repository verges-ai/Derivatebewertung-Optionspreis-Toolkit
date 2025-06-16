# Berechnungen für Preis & Greeks
# pricing/black_scholes.py

import numpy as np
from scipy.stats import norm

def black_scholes_price(S, K, T, r, sigma, option_type='call'):
    d1 = (np.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)

    if option_type == 'call':
        return S * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)
    elif option_type == 'put':
        return K * np.exp(-r * T) * norm.cdf(-d2) - S * norm.cdf(-d1)
    else:
        raise ValueError("option_type must be 'call' or 'put'")

def greeks(S, K, T, r, sigma):
    d1 = (np.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)

    pdf_d1 = norm.pdf(d1)
    cdf_d1 = norm.cdf(d1)
    cdf_d2 = norm.cdf(d2)

    call = {
        'delta': cdf_d1,
        'gamma': pdf_d1 / (S * sigma * np.sqrt(T)),
        'vega': S * pdf_d1 * np.sqrt(T),
        'theta': (-S * pdf_d1 * sigma / (2 * np.sqrt(T)) 
                  - r * K * np.exp(-r * T) * cdf_d2),
        'rho': K * T * np.exp(-r * T) * cdf_d2
    }

    put = {
        'delta': cdf_d1 - 1,
        'gamma': call['gamma'],
        'vega': call['vega'],
        'theta': (-S * pdf_d1 * sigma / (2 * np.sqrt(T)) 
                  + r * K * np.exp(-r * T) * norm.cdf(-d2)),
        'rho': -K * T * np.exp(-r * T) * norm.cdf(-d2)
    }

    return {'call': call, 'put': put}
