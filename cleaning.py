import pandas as pd

# Load the CSV file into a DataFrame
data = pd.read_csv('DISASTERS Data.csv')

# Filter data for disasters in Hong Kong and of type 'flood'
hong_kong_floods = data[(data['Country'] == 'Hong Kong') & (data['Disaster Type'] == 'Flood')]

# Specify the path for the output CSV file
output_file = 'hong_kong_floods.csv'

# Save the filtered data to the output CSV file
hong_kong_floods.to_csv(output_file, index=False)

print("Filtered data has been saved to", output_file)
