import numpy as np
import pandas as pd
from flask import Flask, request, render_template
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
import pickle
import math

# Initialize Flask application
app = Flask(__name__)

# Load and preprocess dataset for predicting weekly rides
data_path = 'uber_dataset.csv'  # Replace with your dataset path
data = pd.read_csv(data_path)

# Calculate 'TotalPrice' based on monthly income, parking charges, and daily expenses
data['TotalPrice'] = data['Monthlyincome'] - (data['Averageparkingpermonth'] * 4) - (data['Dailyexpenses'] * 30)
X = data[['TotalPrice']]
y = data['Rides']

# Split dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train Linear Regression model
model = LinearRegression()
model.fit(X_train, y_train)

# Save the model to disk using pickle
model_file = 'model.pkl'
pickle.dump(model, open(model_file, 'wb'))

# Load and preprocess dataset for predicting ride prices
data_path = 'uber-rides-dataset-indian-standards-time.csv'  # Replace with your dataset path
data = pd.read_csv(data_path)

# Clean data and select relevant features
data['trip_start_time'] = pd.to_datetime(data['trip_start_time'], dayfirst=True)
data['hour'] = data['trip_start_time'].dt.hour
data['day'] = data['trip_start_time'].dt.day
data['month'] = data['trip_start_time'].dt.month

# Select features and target for ride price prediction
features = ['hour', 'day', 'month', 'distance_kms']
target = 'fare_amount'

# Drop rows with missing target values
data = data.dropna(subset=[target])

X = data[features]
y = data[target]

# Split dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train Linear Regression model for ride price prediction
model1 = LinearRegression()
model1.fit(X_train, y_train)

# Save the model to disk using pickle
model1_file = 'model1.pkl'
pickle.dump(model1, open(model1_file, 'wb'))

# Route to home page
@app.route('/')
def home():
    return render_template('home.html')

# Route to form for predicting weekly rides
@app.route('/predict-form')
def predict_form():
    return render_template('index.html')

# Route to form for predicting ride price
@app.route('/predict-ride-form')
def predict_ride_form():
    return render_template('bike-ride.html')

# Route for predicting weekly rides based on income, parking charges, and expenses
@app.route('/predict', methods=['POST'])
def predict():
    try:
        weekly_income = float(request.form['weekly_income'])
        monthly_income = weekly_income * 4
        parking_charges = float(request.form['parking_charges'])
        daily_expenses = float(request.form['daily_expenses'])

        total_price = monthly_income - (parking_charges * 4) - (daily_expenses * 30)

        final_features = np.array([[total_price]])
        prediction = model.predict(final_features)
        output = math.floor(prediction[0])

        return render_template('index.html', prediction_text=f"Number of Rides in the week,it Should be {output},The total profit the driver gains comparing weekely demand ride profit he/she gets {total_price} price per month")
    except ValueError:
        return render_template('index.html', prediction_text="Please enter valid numbers for all inputs.")

# Route for predicting ride price based on date, time, and distance
@app.route('/predict-ride', methods=['POST'])
def predict_ride():
    try:
        date = request.form['date']
        time = request.form['time']
        kilometers = float(request.form['kilometers'])

        datetime_str = f"{date} {time}"
        datetime_obj = pd.to_datetime(datetime_str)

        hour = datetime_obj.hour
        day = datetime_obj.day
        month = datetime_obj.month

        features = np.array([[hour, day, month, kilometers]])
        prediction = model1.predict(features)
        output = round(prediction[0], 2)

        return render_template('bike-ride.html', prediction_text=f"Estimated Ride Price: {output}")
    except ValueError:
        return render_template('bike-ride.html', prediction_text="Please enter valid numbers for all inputs.")

# Run the Flask app
if __name__ == '__main__':
    app.run(debug=True)
