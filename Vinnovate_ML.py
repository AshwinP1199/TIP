import sklearn as sk
import pandas as pd
import numpy as np
from sklearn import preprocessing, model_selection, svm
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
import math

# Load RainData and remove 'Total' column
df = pd.read_excel('RainData.xlsx')
df.drop(columns=['Total'], inplace=True)

# Load FloodDates and set index to 'Year'
df1 = pd.read_excel('FloodDates.xlsx')
df.set_index('Year', inplace=True)

# Extract MonthYear and Rainfall data
month_year = []
rainfall = []
for i in df.index:
    rainfall.append(list(df.loc[i]))
    for j in df.loc[df.index==i]:
        month_year.append(j + f'{i}')

rainfall = np.array(rainfall).flatten()
month_year = np.array(month_year)
data = pd.DataFrame({'MonthYear': month_year, 'Rainfall': rainfall})

# Prepare flood data
df1.sort_values(by='Year', inplace=True)
flooddata = [f"{month}{year}" for month, year in zip(df1['Month'], df1['Year'])]

# Prepare output column
data['output'] = np.nan
c = 0
for i in range(len(data['MonthYear'])):
    if c < len(flooddata):
        if data.at[i, 'MonthYear'] == flooddata[c]:
            data.at[i, 'output'] = 1
            c += 1
        else:
            data.at[i, 'output'] = 0
    else: 
        data = data.fillna(0)
        break

# Prepare features and target
X = np.array(data.drop(columns=['output', 'MonthYear']))
y = np.array(data['output'])

# Split data into training and testing sets
X_train, X_test, y_train, y_test = model_selection.train_test_split(X, y, test_size=0.2)

# Train logistic regression model
clf = LogisticRegression()
clf.fit(X_train, y_train)

# Evaluate model accuracy
acc = clf.score(X_test, y_test)
coeff = clf.coef_
inter = clf.intercept_
print(acc)

# Define sigmoid function for prediction
def sigmoid(val):
    return 1/(1 + math.exp(-(coeff*val + inter/4)))

# Predictions
pred = clf.predict(X_test)
score = sum(1 for i in range(len(y_test)) if y_test[i] == pred[i])

# User input for rainfall prediction
print('Enter mm of rainfall: ')
val = int(input())

acc = score/len(y_test)
print('Prediction: ', round(sigmoid(val)))
print('Prediction confidence: ', sigmoid(val)*100, '%')
