# Step 1: Import Required Libraries
import pandas as pd
from scipy.stats import pearsonr

# Step 2: Load the Dataset
file_path = '/Users/zhenghaozhang/hw/4740/project/final_normalized_data.csv'  # Update path if needed
final_dataset = pd.read_csv(file_path)

# Step 3: Define Numeric Columns for Correlation Analysis
numeric_columns = [
    'Study_Hours_Per_Day',
    'Extracurricular_Hours_Per_Day',
    'Sleep_Hours_Per_Day',
    'Social_Hours_Per_Day',
    'Physical_Activity_Hours_Per_Day',
    'GPA'
]

# Step 4: Perform Correlation Analysis
correlation_results = {}

for column in numeric_columns:
    if column != "GPA":  # GPA is the target variable
        corr, p_value = pearsonr(final_dataset[column], final_dataset["GPA"])
        correlation_results[column] = {"Correlation": corr, "P-Value": p_value}

# Step 5: Convert Results to a DataFrame
correlation_df = pd.DataFrame.from_dict(correlation_results, orient="index").reset_index()
correlation_df.columns = ["Feature", "Correlation", "P-Value"]

# Sort by correlation strength
correlation_df.sort_values(by="Correlation", ascending=False, inplace=True)

# Step 6: Display the Results
correlation_df.reset_index(drop=True, inplace=True)
print("Correlation Analysis Results:")
print(correlation_df)
