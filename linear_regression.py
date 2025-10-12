import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

df = pd.read_csv('mumbai_house_prices.csv')

df = df[df['locality'].str.contains('vikhroli', case=False, na=False)]

df = df.dropna(subset=['price', 'price_unit'])

def convert_price(row):
    try:
        price = float(row['price'])  # ensure it's numeric
        if row['price_unit'].strip().lower() == 'cr':
            return price * 100
        elif row['price_unit'].strip().lower() == 'l':
            return price
        else:
            return None
    except:
        return None
    
df['price_lakhs'] = df.apply(convert_price, axis=1)

df = df.dropna(subset=['price_lakhs'])



df = df[pd.to_numeric(df['area'], errors='coerce').notnull()]
df['area'] = df['area'].astype(float)

X = df['area'].values
Y = df['price_lakhs'].values

X = X.reshape(-1, 1)
Y = Y.reshape(-1, 1)



theta0 = 0.0
theta1 = 0.0

alpha = 0.000001
iterations = 1000

def predict(X, theta0, theta1):
    return theta0 + theta1 * X

def compute_cost(X, Y, theta0, theta1):
    m = len(Y)
    predictions = predict(X, theta0, theta1)
    errors = predictions - Y
    cost = (1 / (2 * m)) * np.sum(errors ** 2)
    return cost

cost_history = []

for i in range(iterations):
    m = len(Y)
    predictions = predict(X, theta0, theta1)
    errors = predictions - Y

    d_theta0 = (1 / m) * np.sum(errors)
    d_theta1 = (1 / m) * np.sum(errors * X)

    theta0 -= alpha * d_theta0
    theta1 -= alpha * d_theta1

    if i % 50 == 0:
        cost = compute_cost(X, Y, theta0, theta1)
        cost_history.append(cost)
        print(f"Iteration: {i}: Cost = {cost: .2f}, θ₀ = {theta0:.2f}, θ₁ = {theta1:.5f}")

y_pred = predict(X, theta0, theta1)

plt.scatter(X, Y, color='blue', label='Actual Data')
plt.plot(X, y_pred, color='red', label='Linear Fit')
plt.xlabel("Area (sq ft)")
plt.ylabel("Price (lakhs Rs.)")
plt.title("House Price Prediction")
plt.legend()
plt.grid(True)
plt.show()

while True:
    try:
        user_area = float(input("Enter area in sq ft (or type -1 to quit): "))
        if user_area == -1:
            break
        predicted_price = theta0 + theta1 * user_area
        print(f"Predicted Price: ₹{predicted_price:.2f} lakhs\n")
    except ValueError:
        print("❌ Invalid input. Please enter a number.\n")