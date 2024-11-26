import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
import pickle

# Load dataset
data_path = 'uber-rides-dataset-indian-standards-time.csv'
data = pd.read_csv(data_path)

# Extracting relevant features
data['trip_start_time'] = pd.to_datetime(data['trip_start_time'],dayfirst=True)
data['hour'] = data['trip_start_time'].dt.hour
data['day'] = data['trip_start_time'].dt.day
data['month'] = data['trip_start_time'].dt.month

# Select features and target
features = ['hour', 'day', 'month', 'distance_kms']
target = 'fare_amount'  # Assuming 'fare_amount' is the target column

# Dropping rows with missing target values
data = data.dropna(subset=[target])

X = data[features]
y = data[target]

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train the model
model = LinearRegression()
model.fit(X_train, y_train)

# Save the model to disk
pickle.dump(model, open('model1.pkl', 'wb'))
