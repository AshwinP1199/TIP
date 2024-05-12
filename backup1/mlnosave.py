import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer

# Load data from the CSV file
df = pd.read_csv('rainfall_data.csv')

# Extract Year, Month, Day, and Rainfall data
X = df[['Year', 'Month', 'Day']]
y = df['Average_Rainfall']

# Convert rainfall measurements into binary labels
y_binary = (y > 0).astype(int)  # 1 for rainfall occurrence, 0 for no rainfall occurrence

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y_binary, test_size=0.2)

# Handle NaN values in input features
imputer = SimpleImputer(strategy='mean')
X_train_imputed = pd.DataFrame(imputer.fit_transform(X_train), columns=X_train.columns)
X_test_imputed = pd.DataFrame(imputer.transform(X_test), columns=X_test.columns)

# Train logistic regression model
clf = LogisticRegression()
clf.fit(X_train_imputed, y_train)

# Evaluate model accuracy
acc = clf.score(X_test_imputed, y_test)
print('Model accuracy:', acc)

# User input for rainfall prediction
print('Enter the year, month, and day: ')
year = int(input('Year: '))
month = int(input('Month (1-12): '))  # Accept numeric input for month
day = int(input('Day: '))

# Handle NaN values in user input
user_input = pd.DataFrame([[year, month, day]], columns=X_train.columns)
user_input_imputed = pd.DataFrame(imputer.transform(user_input), columns=X_test.columns)

# Make prediction
prediction_proba = clf.predict_proba(user_input_imputed)
probability_of_rainfall = prediction_proba[0][1] * 100  # Probability of rainfall occurrence in percentage
print('Probability of rainfall occurrence:', probability_of_rainfall, '%')
