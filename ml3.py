import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

# Load rainfall data from the CSV file
rainfall_data = pd.read_csv('rainfall_data.csv')

# Load flood occurrence data from the CSV file
flood_occurrence = pd.read_csv('flood_occurrence.csv')

# Merge datasets based on Year and Month
merged_data = pd.merge(rainfall_data, flood_occurrence, on=['Year', 'Month'])

# Prepare features (Year, Month, and rainfall data for specific locations) and target (Flood Occurrence)
X = merged_data.drop(['Year', 'Month', 'Day', 'Flood_Occurrence'], axis=1)
y = merged_data['Flood_Occurrence']

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

# Train logistic regression model
clf = LogisticRegression()
clf.fit(X_train, y_train)

# Evaluate model accuracy
accuracy = clf.score(X_test, y_test)
print("Model Accuracy:", accuracy)

# User input for specific day, month, and year
print('Enter the day, month, and year (e.g., 1, 1, 2022): ')
day = int(input('Day: '))
month = int(input('Month: '))
year = int(input('Year: '))

# Locate the corresponding row in the dataset
input_data = merged_data[(merged_data['Year'] == year) & (merged_data['Month'] == month) & (merged_data['Day'] == day)]

# Make prediction for flood occurrence probability
if not input_data.empty:
    input_features = input_data.drop(['Year', 'Month', 'Day', 'Flood_Occurrence'], axis=1)
    probability = clf.predict_proba(input_features)[0][1] * 100  # Probability of flood occurrence in percentage
    print('Probability of Flood Occurrence (in percentage):', probability)
else:
    print('No data available for the specified date.')
