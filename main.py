from pricing.black_scholes import black_scholes_price, greeks

S = 100
K = 100
T = 1
r = 0.02
sigma = 0.25

price = black_scholes_price(S, K, T, r, sigma, 'call')
g = greeks(S, K, T, r, sigma)['call']

print(f"Preis: {price:.2f}")
print("Greeks:", g)
