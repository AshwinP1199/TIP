import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
import math

# Load data from the CSV file
df = pd.read_csv('Average Hong Kong Rainfall.csv')

# Extract features (Year, Month, Day, and Average Rainfall)
X = df[['Year', 'Month', 'Day', 'Average_Rainfall']]

# Define target variable: 1 for flood, 0 for no flood
# Assuming that if the average rainfall exceeds a certain threshold, it leads to flooding
threshold_rainfall = 100  # Adjust the threshold as needed
df['Flood'] = np.where(df['Average_Rainfall'] > threshold_rainfall, 1, 0)
y = df['Flood']

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

# Train logistic regression model
clf = LogisticRegression()
clf.fit(X_train, y_train)

# Evaluate model accuracy
acc = clf.score(X_test, y_test)
print("Model Accuracy:", acc)

# User input for rainfall prediction
print('Enter the year, month, day, and average rainfall: ')
year = int(input('Year: '))
month = int(input('Month (1-12): '))
day = int(input('Day: '))
average_rainfall = float(input('Average Rainfall: '))

# Make prediction
prediction_proba = clf.predict_proba([[year, month, day, average_rainfall]])[:, 1]
print('Probability of flood:', prediction_proba[0] * 100, '%')
