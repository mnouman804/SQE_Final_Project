import pandas as pd
from deepchecks.tabular import Dataset
from deepchecks.tabular.suites import data_integrity

# Sample dataset with a date column
data = {
    'name': ['Alice', 'Bob', 'Charlie', 'Bob', 'Eve'],
    'age': [25, 30, None, 30, 22],
    'gender': ['F', 'M', 'O', 'M', 'F'],
    'income': [50000, 54000, 52000, 54000, None],
    'joined_date': ['2022-01-01', '2021-12-01', '2023-01-01', '2021-12-01', '2022-05-15']
}

# Convert 'joined_date' to datetime format
df = pd.DataFrame(data)
df['joined_date'] = pd.to_datetime(df['joined_date'])

# Define dataset schema (categorical features only)
dataset = Dataset(df, cat_features=['gender'])

# Run a comprehensive Data Integrity suite
suite = data_integrity()
suite_result = suite.run(dataset)

# Save the entire suite report as an HTML file
suite_result.save_as_html('deepchecks_advanced_report.html')

print("Advanced Data Integrity report saved as deepchecks_advanced_report.html")
