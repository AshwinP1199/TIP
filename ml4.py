import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.impute import SimpleImputer

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

# Create and open a new CSV file to save predictions
with open('pred.csv', mode='w') as file:
    # Write the header line
    file.write("Year,Month,Day,Prediction %\n")
    
    # Loop through years from 2010 to 2023
    for year in range(2010, 2024):
        # Loop through months (1-12)
        for month in range(1, 13):
            # Loop through days (1-31)
            for day in range(1, 32):
                # Handle leap years and months with fewer than 31 days
                if month in [4, 6, 9, 11] and day == 31:
                    continue
                if month == 2 and day > 28:
                    if (year % 4 == 0 and year % 100 != 0) or (year % 400 == 0):
                        if day > 29:
                            continue
                    else:
                        continue
                
                # Make prediction for the current date
                user_input = pd.DataFrame([[year, month, day]], columns=X.columns)
                user_input_imputed = pd.DataFrame(imputer.transform(user_input), columns=X.columns)
                prediction_proba = clf.predict_proba(user_input_imputed)
                probability_of_rainfall = prediction_proba[0][1] * 100  # Probability of rainfall occurrence in percentage
                
                # Save prediction to the CSV file
                file.write(f"{year},{month},{day},{probability_of_rainfall}\n")
