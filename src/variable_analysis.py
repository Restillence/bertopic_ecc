# read csv file from filepath
import pandas as pd
filepath = "D:/daten_masterarbeit/merged_topics_crsp_sample.csv"
df = pd.read_csv(filepath, index_col=0)

#%% data preparation
import pandas as pd
import numpy as np

# Assume 'merged_df' is your DataFrame
# Remove rows with missing values in the relevant columns
analysis_df = df[['similarity_to_average', 'ret_next_day', 'ret_5_days', 'ret_20_days', 'ret_60_days']].dropna()

# Display the number of observations
print(f"Number of observations after dropping NaNs: {len(analysis_df)}")


#%% exploratory data analysis
import matplotlib.pyplot as plt
import seaborn as sns

# Set up the plotting style
sns.set_style('whitegrid')

# List of return variables
return_vars = ['ret_next_day', 'ret_5_days', 'ret_20_days', 'ret_60_days', 'epsfxq']

# Create scatter plots
for ret_var in return_vars:
    plt.figure(figsize=(8, 6))
    sns.scatterplot(data=analysis_df, x='similarity_to_average', y=ret_var, alpha=0.5)
    plt.title(f'Scatter Plot of {ret_var} vs. Similarity to Average')
    plt.xlabel('Similarity to Average')
    plt.ylabel(f'{ret_var}')
    plt.show()

#%% correlations
# Calculate correlation coefficients
correlations = analysis_df.corr(method='pearson')

# Extract correlations with 'similarity_to_average'
similarity_correlations = correlations['similarity_to_average'][return_vars]

print("Correlation coefficients between 'similarity_to_average' and return variables:")
print(similarity_correlations)


#%% linear regression
import statsmodels.api as sm

# Function to perform regression and display results
def perform_regression(y_var):
    X = analysis_df[['similarity_to_average']]
    y = analysis_df[y_var]
    X = sm.add_constant(X)  # Adds a constant term to the predictor
    model = sm.OLS(y, X).fit()
    print(f"Regression results for {y_var} ~ similarity_to_average:")
    print(model.summary())
    print("\n")

# Perform regression for each return variable
for ret_var in return_vars:
    perform_regression(ret_var)

#%% linear regression sklearn
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score

# Function to perform regression and display results
def perform_sklearn_regression(y_var):
    X = analysis_df[['similarity_to_average']].values
    y = analysis_df[y_var].values
    model = LinearRegression()
    model.fit(X, y)
    y_pred = model.predict(X)
    r2 = r2_score(y, y_pred)
    print(f"Regression results for {y_var} ~ similarity_to_average:")
    print(f"Coefficient: {model.coef_[0]}")
    print(f"Intercept: {model.intercept_}")
    print(f"R-squared: {r2}")
    print("\n")

# Perform regression for each return variable
for ret_var in return_vars:
    perform_sklearn_regression(ret_var)

#%% other plots
# Plot histogram and density of 'similarity_to_average'
plt.figure(figsize=(8, 6))
sns.histplot(analysis_df['similarity_to_average'], kde=True)
plt.title('Distribution of Similarity to Average')
plt.xlabel('Similarity to Average')
plt.ylabel('Frequency')
plt.show()

