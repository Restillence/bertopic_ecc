# read csv file from filepath
import pandas as pd
filepath = "D:/daten_masterarbeit/final_dataset.csv"
df = pd.read_csv(filepath, index_col=0)

print(f"Number of observations in the final_dataset: {len(df)}")

#%% data preparation
import numpy as np

# Updated list of variables to include in the analysis
variables = [
    'similarity_to_overall_average',
    'similarity_to_industry_average',
    'similarity_to_company_average',
    'excess_ret',
    'excess_ret_next_day',
    'excess_ret_5_days',
    'excess_ret_20_days',
    'excess_ret_60_days',
    'epsfxq',
    'epsfxq_next'
]

# Create analysis DataFrame with the specified variables
analysis_df = df[variables].dropna()

# Display the number of observations
print(f"Number of observations after dropping NaNs: {len(analysis_df)}")

#%% exploratory data analysis
import matplotlib.pyplot as plt
import seaborn as sns

# Set up the plotting style
sns.set_style('whitegrid')

# Updated list of return variables
return_vars = [
    'excess_ret',
    'excess_ret_next_day',
    'excess_ret_5_days',
    'excess_ret_20_days',
    'excess_ret_60_days',
    'epsfxq',
    'epsfxq_next'
]

# List of similarity variables (unchanged)
similarity_vars = [
    'similarity_to_overall_average',
    'similarity_to_industry_average',
    'similarity_to_company_average'
]
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
#%% correlations
# Calculate correlation coefficients
correlations = analysis_df.corr(method='pearson')

# Display the correlation matrix
print("Correlation matrix:")
print(correlations)

# Extract correlations of similarity variables with return variables
for sim_var in similarity_vars:
    print(f"\nCorrelation coefficients between '{sim_var}' and return variables:")
    print(correlations[sim_var][return_vars])

#%% linear regression using statsmodels
import statsmodels.api as sm

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
    model = perform_regression_with_statsmodels(ret_var, similarity_vars)


#%% linear regression using scikit-learn
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_squared_error

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
    model = perform_sklearn_regression(ret_var, similarity_vars)

#%% additional plots
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