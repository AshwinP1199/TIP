import pandas as pd
import numpy as np
import os

# Function to read and merge CSV files
def merge_csv_files(folder_path):
    all_data = pd.DataFrame()  # Initialize an empty DataFrame to store merged data
    
    # Loop through all CSV files in the folder
    for file in os.listdir(folder_path):
        if file.endswith(".csv"):
            file_path = os.path.join(folder_path, file)
            df = pd.read_csv(file_path)  # Read CSV file into a DataFrame
            location = os.path.splitext(file)[0]  # Extract location name from file name
            
            # Rename the 'Value' column to match the location
            df.rename(columns={"Value": location}, inplace=True)
            
            # Merge data into the main DataFrame
            all_data = pd.concat([all_data, df[['Year', 'Month', 'Day', location]]], ignore_index=True)
    
    # Pivot the DataFrame to have locations as columns
    all_data_pivoted = all_data.pivot_table(index=['Year', 'Month', 'Day'], aggfunc='first').reset_index()
    
    # Convert all columns except 'Year', 'Month', 'Day' to numeric
    all_data_pivoted.iloc[:, 3:] = all_data_pivoted.iloc[:, 3:].apply(pd.to_numeric, errors='coerce')
    
    # Calculate the average rainfall for each day
    all_data_pivoted['Average_Rainfall'] = all_data_pivoted.iloc[:, 3:].mean(axis=1, skipna=True)
    
    # Write the merged DataFrame to a new CSV file
    all_data_pivoted.to_csv("merged_rainfall_data_with_average.csv", index=False)

# Specify the folder containing the CSV files
folder_path = r"C:\Users\ashwi\Downloads\test2"

# Call the function to merge CSV files
merge_csv_files(folder_path)
