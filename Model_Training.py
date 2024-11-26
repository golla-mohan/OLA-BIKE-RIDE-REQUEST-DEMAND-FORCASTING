import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
import pickle

# Load dataset
# Replace 'your_dataset.csv' with the path to your dataset
data = pd.read_csv('uber_dataset.csv')

# Assuming your dataset has the following columns: 'TotalPrice' and 'Rides'
data['TotalPrice'] = data['Monthlyincome'] - (data['Averageparkingpermonth'] * 4) - (data['Dailyexpenses'] * 30)
X = data[['TotalPrice']]
y = data['Rides']

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train the model
model = LinearRegression()
model.fit(X_train, y_train)

# Save the model to disk
pickle.dump(model, open('model.pkl', 'wb'))

