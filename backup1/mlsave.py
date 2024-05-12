import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.impute import SimpleImputer
import os

# Load data from the CSV file
df = pd.read_csv('rainfall_data.csv')

# Extract Year, Month, Day, and Rainfall data
X = df[['Year', 'Month', 'Day']]
y = df['Average_Rainfall']

# Convert rainfall measurements into binary labels
y_binary = (y > 0).astype(int)  # 1 for rainfall occurrence, 0 for no rainfall occurrence

# Handle NaN values in input features
imputer = SimpleImputer(strategy='mean')
X_imputed = pd.DataFrame(imputer.fit_transform(X), columns=X.columns)

# Train logistic regression model
clf = LogisticRegression()
clf.fit(X_imputed, y_binary)

# Check if the predictions file exists, if not, create it
if not os.path.exists('predictions.csv'):
    with open('predictions.csv', 'w') as file:
        file.write('Year,Month,Day,Prediction %\n')

# User input for rainfall prediction
year = int(input('Enter the year: '))
month = int(input('Enter the month (1-12): '))
day = int(input('Enter the day: '))

# Handle NaN values in user input
user_input = pd.DataFrame([[year, month, day]], columns=X.columns)
user_input_imputed = pd.DataFrame(imputer.transform(user_input), columns=X.columns)

# Make prediction
prediction_proba = clf.predict_proba(user_input_imputed)
probability_of_rainfall = prediction_proba[0][1] * 100  # Probability of rainfall occurrence in percentage

# Append prediction to the CSV file
with open('predictions.csv', mode='a+') as file:
    # Check if the file is empty
    file.seek(0)
    first_char = file.read(1)
    if not first_char:
        file.write('Year,Month,Day,Prediction %\n')  # Write header if file is empty
    file.write(f"{year},{month},{day},{probability_of_rainfall}\n")

print('Probability of rainfall occurrence:', probability_of_rainfall, '%')
print("Prediction saved to 'predictions.csv'.")
