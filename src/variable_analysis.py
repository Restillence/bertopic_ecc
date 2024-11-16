# read csv file from filepath
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import statsmodels.api as sm
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_squared_error

# Path to the final dataset
filepath = "D:/daten_masterarbeit/final_dataset.csv"
filepath = "D:/daten_masterarbeit/final_dataset_2500_reg_20.csv"

# Read the CSV file, assuming 'call_id' is the index column if applicable
df = pd.read_csv(filepath, index_col=0)

print(f"Number of observations in the final_dataset: {len(df)}")

#%% Data Preparation

# Updated list of variables to include in the analysis
variables = [
    'similarity_to_overall_average',
    'similarity_to_industry_average',
    'similarity_to_company_average',
    'excess_ret_immediate',
    'excess_ret_short_term',
    'excess_ret_medium_term',
    'excess_ret_long_term',
    'epsfxq',
    'epsfxq_next'
]

# Create analysis DataFrame with the specified variables
analysis_df = df[variables].dropna()

# Display the number of observations
print(f"Number of observations after dropping NaNs: {len(analysis_df)}")

#%% Exploratory Data Analysis

# Set up the plotting style
sns.set_style('whitegrid')

# Updated list of return variables
return_vars = [
    'excess_ret_immediate',
    'excess_ret_short_term',
    'excess_ret_medium_term',
    'excess_ret_long_term',
    'epsfxq',
    'epsfxq_next'
]

# List of similarity variables (unchanged)
similarity_vars = [
    'similarity_to_overall_average',
    'similarity_to_industry_average',
    'similarity_to_company_average'
]

# Uncomment the following block to create scatter plots
"""
# Create scatter plots between similarity variables and return variables
for sim_var in similarity_vars:
    for ret_var in return_vars:
        plt.figure(figsize=(8, 6))
        sns.scatterplot(data=analysis_df, x=sim_var, y=ret_var, alpha=0.5)
        plt.title(f'Scatter Plot of {ret_var} vs. {sim_var}')
        plt.xlabel(f'{sim_var}')
        plt.ylabel(f'{ret_var}')
        plt.show()

# Plot histograms and density plots for similarity variables
for sim_var in similarity_vars:
    plt.figure(figsize=(8, 6))
    sns.histplot(analysis_df[sim_var], kde=True)
    plt.title(f'Distribution of {sim_var}')
    plt.xlabel(sim_var)
    plt.ylabel('Frequency')
    plt.show()
"""

#%% Correlations

# Calculate correlation coefficients
correlations = analysis_df.corr(method='pearson')

# Display the correlation matrix
print("Correlation matrix:")
print(correlations)

# Extract correlations of similarity variables with return variables
for sim_var in similarity_vars:
    print(f"\nCorrelation coefficients between '{sim_var}' and return variables:")
    print(correlations.loc[sim_var, return_vars])

#%% Linear Regression using Statsmodels

# Function to perform regression with statsmodels and display p-values
def perform_regression_with_statsmodels(y_var, x_vars):
    X = analysis_df[x_vars]
    y = analysis_df[y_var]
    X = sm.add_constant(X)  # Adds a constant term to the predictor
    model = sm.OLS(y, X).fit()
    print(f"Regression results for {y_var} ~ {', '.join(x_vars)}:")
    print(model.summary())  # This will include p-values, R-squared, and other stats
    print("\n")
    return model

# Perform regression for each return variable on similarity variables
for ret_var in return_vars:
    perform_regression_with_statsmodels(ret_var, similarity_vars)

#%% Linear Regression using Scikit-learn

# Function to perform regression and display results
def perform_sklearn_regression(y_var, x_vars):
    X = analysis_df[x_vars].values
    y = analysis_df[y_var].values
    model = LinearRegression()
    model.fit(X, y)
    y_pred = model.predict(X)
    r2 = r2_score(y, y_pred)
    mse = mean_squared_error(y, y_pred)
    print(f"Regression results for {y_var} ~ {', '.join(x_vars)}:")
    print(f"Coefficients: {dict(zip(x_vars, model.coef_))}")
    print(f"Intercept: {model.intercept_}")
    print(f"R-squared: {r2}")
    print(f"Mean Squared Error: {mse}")
    print("\n")
    return model

# Perform regression of return variables on similarity variables
for ret_var in return_vars:
    perform_sklearn_regression(ret_var, similarity_vars)

#%% Additional Plots

"""
# Plot histograms and density of 'epsfxq' and 'epsfxq_next'
for eps_var in ['epsfxq', 'epsfxq_next']:
    plt.figure(figsize=(8, 6))
    sns.histplot(analysis_df[eps_var], kde=True)
    plt.title(f'Distribution of {eps_var}')
    plt.xlabel(eps_var)
    plt.ylabel('Frequency')
    plt.show()

# Scatter plots between EPS variables and similarity variables
for sim_var in similarity_vars:
    for eps_var in ['epsfxq', 'epsfxq_next']:
        plt.figure(figsize=(8, 6))
        sns.scatterplot(data=analysis_df, x=sim_var, y=eps_var, alpha=0.5)
        plt.title(f'Scatter Plot of {eps_var} vs. {sim_var}')
        plt.xlabel(f'{sim_var}')
        plt.ylabel(f'{eps_var}')
        plt.show()
"""
