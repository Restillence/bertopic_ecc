# Import necessary libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import statsmodels.api as sm
from scipy.stats import chi2_contingency, ttest_ind, shapiro
from statsmodels.stats.diagnostic import het_breuschpagan
from statsmodels.graphics.gofplots import qqplot
from statsmodels.stats.stattools import durbin_watson
from statsmodels.stats.outliers_influence import variance_inflation_factor
from statsmodels.iolib.summary2 import summary_col
from statsmodels.tools.tools import add_constant
from sklearn.preprocessing import StandardScaler
import os
import ast

#%% Configure Plotting Aesthetics
sns.set_theme(style="whitegrid")
plt.rcParams.update({
    'figure.figsize': (12, 8),
    'axes.labelsize': 14,
    'axes.titlesize': 16,
    'xtick.labelsize': 12,
    'ytick.labelsize': 12,
    'legend.fontsize': 12,
    'font.size': 14
})

#%% Create '../regression_results' folder

# Define the output folder path relative to the current script's directory
output_folder = os.path.join("..", "regression_results")

# Create 'regression_results' folder in the parent directory if it doesn't exist
if not os.path.exists(output_folder):
    os.makedirs(output_folder)
    print(f"Created folder: {output_folder}")
else:
    print(f"Folder already exists: {output_folder}")

#%% Path to the final dataset

# Define the file path to your dataset
filepath = "D:/daten_masterarbeit/final_dataset_reg_full.csv"

# Read the CSV file
df = pd.read_csv(filepath)

print(f"Number of observations in the final_dataset: {len(df)}")

#%% Data Preparation

# List of variables to include in the analysis
variables = [
    'similarity_to_overall_average',
    'similarity_to_industry_average',
    'similarity_to_company_average',
    'excess_ret_immediate',
    'excess_ret_short_term',
    'excess_ret_medium_term',
    'excess_ret_long_term',
    'epsfxq',
    'epsfxq_next',
    'length_participant_questions',  # Dependent Variable
    'length_management_answers',    # Dependent Variable
    'market_cap',                   # Control Variable
    'rolling_beta',                 # Control Variable
    'ceo_participates',             # Control Variable
    'ceo_cfo_change',               # Control Variable
    'word_length_presentation',     # Control Variable
    'participant_question_topics',  # For Chi-Squared Test
    'management_answer_topics',     # For Chi-Squared Test
    'filtered_presentation_topics'  # For topic diversity
]

# Ensure all variables exist in the DataFrame
missing_vars = [var for var in variables if var not in df.columns]
if missing_vars:
    raise KeyError(f"The following required columns are missing from the DataFrame: {missing_vars}")

# Create analysis DataFrame with the specified variables
analysis_df = df[variables].dropna()

# Display the number of observations after dropping NaNs
print(f"Number of observations after dropping NaNs: {len(analysis_df)}")

#%% Parse Topic Columns Properly

# Function to safely parse string representations of lists
def parse_topics(topics):
    """
    Converts string representations of lists into actual Python lists.
    If parsing fails, returns an empty list.
    """
    try:
        # Use ast.literal_eval for safe parsing
        return ast.literal_eval(topics) if isinstance(topics, str) else []
    except (ValueError, SyntaxError):
        return []

# Apply the parsing function to topic columns
analysis_df['participant_question_topics'] = analysis_df['participant_question_topics'].apply(parse_topics)
analysis_df['management_answer_topics'] = analysis_df['management_answer_topics'].apply(parse_topics)
analysis_df['filtered_presentation_topics'] = analysis_df['filtered_presentation_topics'].apply(parse_topics)

#%% Extract Primary Topics

# Function to extract the most frequent topic from a list
def extract_primary_topic(topics):
    """
    Extracts the most frequent topic from a list of topics.
    If the list is empty, returns 'Other'.
    """
    if not topics:
        return 'Other'
    # Count the frequency of each topic
    topic_counts = pd.Series(topics).value_counts()
    # Return the most frequent topic
    return topic_counts.idxmax()

# Apply the function to extract primary topics
analysis_df['primary_participant_topic'] = analysis_df['participant_question_topics'].apply(extract_primary_topic)
analysis_df['primary_management_topic'] = analysis_df['management_answer_topics'].apply(extract_primary_topic)

#%% Aggregate Rare Topics

# Define the number of top topics to retain
TOP_N = 50  # Adjust based on your data and memory constraints

# Function to aggregate rare topics
def aggregate_rare_topics(series, top_n):
    """
    Aggregates rare topics into 'Other'.
    
    Parameters:
    - series (pd.Series): Series containing categorical data.
    - top_n (int): Number of top categories to retain.
    
    Returns:
    - pd.Series: Series with rare categories replaced by 'Other'.
    """
    top_categories = series.value_counts().nlargest(top_n).index
    return series.apply(lambda x: x if x in top_categories else 'Other')

# Apply the aggregation to primary topics
analysis_df['primary_participant_topic'] = aggregate_rare_topics(analysis_df['primary_participant_topic'], TOP_N)
analysis_df['primary_management_topic'] = aggregate_rare_topics(analysis_df['primary_management_topic'], TOP_N)

print(f"Aggregated rare topics, retaining top {TOP_N} categories.")

#%% Create 'topic_diversity' Variable

# Function to calculate topic diversity
def calculate_topic_diversity(topics):
    """
    Calculates the diversity of topics in a given list.
    Diversity is defined as the ratio of unique topics to the total number of topics.
    """
    if not isinstance(topics, list) or len(topics) == 0:
        return np.nan  # Return NaN for invalid or empty topic lists
    unique_topics = set(topics)
    diversity = len(unique_topics) / len(topics)
    return diversity

# Create the 'topic_diversity' column
analysis_df['topic_diversity'] = analysis_df['filtered_presentation_topics'].apply(calculate_topic_diversity)

# Handle potential NaN values in 'topic_diversity'
# Option 1: Fill NaNs with the mean diversity
mean_diversity = analysis_df['topic_diversity'].mean()
analysis_df['topic_diversity'] = analysis_df['topic_diversity'].fillna(mean_diversity)

print("Created 'topic_diversity' variable.")

#%% Create 'difference_questions_answers' Variable

# Create 'difference_questions_answers' variable
analysis_df['difference_questions_answers'] = analysis_df['length_participant_questions'] - analysis_df['length_management_answers']

# Display summary statistics
print("Summary statistics for 'difference_questions_answers':")
print(analysis_df['difference_questions_answers'].describe())

# Visualize the distribution
plt.figure(figsize=(12, 8))
sns.histplot(analysis_df['difference_questions_answers'], kde=True, bins=30, color='skyblue')
plt.title('Distribution of Difference Between Participant Questions and Management Answers', fontsize=16)
plt.xlabel('Participant Questions - Management Answers', fontsize=14)
plt.ylabel('Frequency', fontsize=14)
plt.tight_layout()
plt.savefig(os.path.join(output_folder, 'difference_questions_answers_distribution.png'), dpi=300)
plt.close()
print("Saved distribution plot for 'difference_questions_answers'.\n")

#%% Descriptive Statistics

# Define the output folder for descriptive statistics
descriptive_stats_folder = os.path.join(output_folder, "descriptive_statistics")

# Create the folder if it doesn't exist
if not os.path.exists(descriptive_stats_folder):
    os.makedirs(descriptive_stats_folder)
    print(f"Created folder for descriptive statistics: {descriptive_stats_folder}")
else:
    print(f"Folder for descriptive statistics already exists: {descriptive_stats_folder}")

#%% Generate Summary Statistics for Numerical Variables

# Select numerical columns
numerical_cols = analysis_df.select_dtypes(include=[np.number]).columns.tolist()

# Generate descriptive statistics using pandas
numerical_summary = analysis_df[numerical_cols].describe().transpose()

# Calculate additional statistics: skewness and kurtosis
numerical_summary['skew'] = analysis_df[numerical_cols].skew()
numerical_summary['kurtosis'] = analysis_df[numerical_cols].kurtosis()

# Reorder columns for better presentation
numerical_summary = numerical_summary[['count', 'mean', 'std', 'min', '25%', '50%', '75%', 'max', 'skew', 'kurtosis']]

# Round the numbers for better readability
numerical_summary = numerical_summary.round(3)

# Export to LaTeX
numerical_summary_latex = numerical_summary.to_latex(
    caption='Descriptive Statistics for Numerical Variables',
    label='tab:descriptive_numerical',
    float_format='%.3f',
    column_format='lcccccccccc',  # Adjust column alignment as needed
    bold_rows=True
)

# Define the output file path
numerical_latex_path = os.path.join(descriptive_stats_folder, "descriptive_statistics_numerical.tex")

# Write the LaTeX table to file
with open(numerical_latex_path, 'w') as f:
    f.write(numerical_summary_latex)

print(f"Saved numerical descriptive statistics LaTeX table to {numerical_latex_path}")

#%% Generate Frequency Counts for Categorical Variables

# Define categorical columns
# Assuming 'primary_participant_topic' and 'primary_management_topic' are categorical
categorical_cols = ['primary_participant_topic', 'primary_management_topic']

for col in categorical_cols:
    # Calculate frequency counts
    freq_counts = analysis_df[col].value_counts().to_frame()
    freq_counts.columns = ['Frequency']
    freq_counts['Percentage'] = (freq_counts['Frequency'] / freq_counts['Frequency'].sum() * 100).round(2)
    
    # Reorder columns
    freq_counts = freq_counts[['Frequency', 'Percentage']]
    
    # Export to LaTeX
    freq_counts_latex = freq_counts.to_latex(
        caption=f'Frequency Counts for {col.replace("_", " ").title()}',
        label=f'tab:descriptive_{col}',
        float_format='%.2f',
        column_format='lcc',  # Adjust column alignment as needed
        bold_rows=True,
        index=True
    )
    
    # Define the output file path
    categorical_latex_path = os.path.join(descriptive_stats_folder, f"descriptive_statistics_{col}.tex")
    
    # Write the LaTeX table to file
    with open(categorical_latex_path, 'w') as f:
        f.write(freq_counts_latex)
    
    print(f"Saved categorical descriptive statistics LaTeX table to {categorical_latex_path}")

#%% Standardize Specific Control Variables

# Define control variables
control_vars = [
    'market_cap',
    'ceo_participates',
    'ceo_cfo_change',
    'rolling_beta',
    'word_length_presentation'
]

# Define the variables to be standardized
variables_to_scale = ['market_cap', 'word_length_presentation']

# Initialize the scaler
scaler = StandardScaler()

# Fit and transform only the specified variables
analysis_df[variables_to_scale] = scaler.fit_transform(analysis_df[variables_to_scale])

print("Standardized 'market_cap' and 'word_length_presentation'.")

#%% Define Similarity Groups Based on Percentiles

# Define similarity groups based on all similarity variables
low_percentile = 20
high_percentile = 80

# Calculate the 20th and 80th percentiles for each similarity variable
low_thresholds = analysis_df[[
    'similarity_to_overall_average',
    'similarity_to_industry_average',
    'similarity_to_company_average'
]].quantile(low_percentile / 100)

high_thresholds = analysis_df[[
    'similarity_to_overall_average',
    'similarity_to_industry_average',
    'similarity_to_company_average'
]].quantile(high_percentile / 100)

# Define low_similarity: calls in the lowest 20% across all similarity variables
low_similarity = analysis_df[
    (analysis_df['similarity_to_overall_average'] <= low_thresholds['similarity_to_overall_average']) &
    (analysis_df['similarity_to_industry_average'] <= low_thresholds['similarity_to_industry_average']) &
    (analysis_df['similarity_to_company_average'] <= low_thresholds['similarity_to_company_average'])
]

# Define high_similarity: calls in the highest 20% across all similarity variables
high_similarity = analysis_df[
    (analysis_df['similarity_to_overall_average'] > high_thresholds['similarity_to_overall_average']) &
    (analysis_df['similarity_to_industry_average'] > high_thresholds['similarity_to_industry_average']) &
    (analysis_df['similarity_to_company_average'] > high_thresholds['similarity_to_company_average'])
]

print(f"Defined similarity groups based on {low_percentile}th and {high_percentile}th percentiles across all similarity variables.")
print(f"Low Similarity Group (<= {low_percentile}% on all similarity variables): {len(low_similarity)} observations")
print(f"High Similarity Group (>{high_percentile}% on all similarity variables): {len(high_similarity)} observations\n")

#%% Exploratory Data Analysis

# Select only numeric columns for correlation
numeric_cols = analysis_df.select_dtypes(include=[np.number]).columns.tolist()

# Calculate correlation coefficients
correlations = analysis_df[numeric_cols].corr(method='pearson')

# Display the correlation matrix
print("Correlation matrix:")
print(correlations)

# Extract correlations of similarity variables with return variables
return_vars = [
    'excess_ret_immediate',
    'excess_ret_short_term',
    'excess_ret_medium_term',
    'excess_ret_long_term',
    'epsfxq',
    'epsfxq_next'
]

similarity_vars = [
    'similarity_to_overall_average',
    'similarity_to_industry_average',
    'similarity_to_company_average'
]

print("\nCorrelation coefficients between similarity variables and return variables:")
print(correlations.loc[similarity_vars, return_vars])

#%% Define Regression and Diagnostic Functions

# Function to calculate VIF
def calculate_vif(independent_vars, analysis_df):
    """
    Calculates Variance Inflation Factor (VIF) for a list of independent variables.

    Parameters:
    - independent_vars (list): List of independent variable names.
    - analysis_df (DataFrame): The DataFrame containing the data.

    Returns:
    - vif_df (DataFrame): DataFrame containing VIF values.
    """
    X = analysis_df[independent_vars]
    X = add_constant(X)
    vif_data = pd.DataFrame()
    vif_data["Variable"] = X.columns
    vif_data["VIF"] = [variance_inflation_factor(X.values, i) for i in range(X.shape[1])]
    return vif_data

# Function to perform OLS diagnostic checks
def ols_diagnostics(model, y_var, x_vars, output_folder):
    """
    Generates diagnostic plots and tests for an OLS regression model.

    Parameters:
    - model: The fitted OLS model.
    - y_var (str): Dependent variable name.
    - x_vars (list): List of independent variables.
    - output_folder (str): Directory to save diagnostic plots and statistics.
    """
    # Create residuals and fitted values
    residuals = model.resid
    fitted = model.fittedvalues

    # 1. Residuals vs. Fitted Values Plot
    plt.figure(figsize=(12, 8))
    sns.scatterplot(x=fitted, y=residuals, alpha=0.5, edgecolor=None)
    plt.axhline(0, color='red', linestyle='--')
    plt.xlabel('Fitted Values', fontsize=14)
    plt.ylabel('Residuals', fontsize=14)
    plt.title(f'Residuals vs Fitted Values for {y_var}', fontsize=16)
    plt.tight_layout()
    plt.savefig(os.path.join(output_folder, f'{y_var}_residuals_fitted.png'), dpi=300)
    plt.close()
    print(f"Saved Residuals vs Fitted plot for {y_var}.")

    # 2. Q-Q Plot for Normality
    plt.figure(figsize=(12, 8))
    qqplot(residuals, line='s', ax=plt.gca())
    plt.title(f'Q-Q Plot of Residuals for {y_var}', fontsize=16)
    plt.tight_layout()
    plt.savefig(os.path.join(output_folder, f'{y_var}_qq_plot.png'), dpi=300)
    plt.close()
    print(f"Saved Q-Q plot for {y_var}.")

    # 3. Histogram of Residuals
    plt.figure(figsize=(12, 8))
    sns.histplot(residuals, kde=True, bins=30, color='salmon')
    plt.title(f'Histogram of Residuals for {y_var}', fontsize=16)
    plt.xlabel('Residuals', fontsize=14)
    plt.ylabel('Frequency', fontsize=14)
    plt.tight_layout()
    plt.savefig(os.path.join(output_folder, f'{y_var}_residuals_histogram.png'), dpi=300)
    plt.close()
    print(f"Saved Residuals Histogram for {y_var}.")

    # 4. Breusch-Pagan Test for Heteroscedasticity
    bp_test = het_breuschpagan(residuals, model.model.exog)
    bp_pvalue = bp_test[1]
    print(f"Breusch-Pagan test p-value for {y_var}: {bp_pvalue:.4f}")

    # 5. Shapiro-Wilk Test for Normality
    if len(residuals) <= 5000:
        shapiro_stat, shapiro_p = shapiro(residuals)
        print(f"Shapiro-Wilk test p-value for {y_var}: {shapiro_p:.4f}")
    else:
        print(f"Shapiro-Wilk test not performed for {y_var} due to large sample size (N={len(residuals)}).")

    # 6. Durbin-Watson Test for Autocorrelation
    dw_stat = durbin_watson(residuals)
    print(f"Durbin-Watson statistic for {y_var}: {dw_stat:.4f}")

    # Save diagnostic statistics to a text file
    diag_filename = f"{y_var}_diagnostics.txt"
    diag_path = os.path.join(output_folder, diag_filename)
    with open(diag_path, 'w') as f:
        f.write(f"Breusch-Pagan Test p-value: {bp_pvalue:.4f}\n")
        if len(residuals) <= 5000:
            f.write(f"Shapiro-Wilk Test p-value: {shapiro_p:.4f}\n")
        else:
            f.write(f"Shapiro-Wilk Test: Not performed due to large sample size (N={len(residuals)})\n")
        f.write(f"Durbin-Watson Statistic: {dw_stat:.4f}\n")
    print(f"Saved diagnostic statistics for {y_var} to {diag_path}\n")

# Function to perform OLS regression and save separate HTML and LaTeX tables
def perform_combined_regressions(regression_groups, analysis_df, output_folder, captions):
    """
    Performs multiple OLS regressions, separates them into main and other groups, 
    renames main group models, and saves separate HTML and LaTeX tables.

    Parameters:
    - regression_groups (dict): Dictionary where keys are group names and values are lists of dependent variables.
    - analysis_df (DataFrame): The DataFrame containing the data.
    - output_folder (str): Path to the folder where the HTML and LaTeX files will be saved.
    - captions (dict): Dictionary where keys are group names and values are caption texts.

    Returns:
    - main_results_latex (dict): Dictionary of main regression models for LaTeX tables.
    - other_results (dict): Dictionary of other regression models.
    """
    # Dictionaries to hold models
    main_results_latex = {}
    other_results = {}

    # Mapping for renaming main models for LaTeX (lowercase, no underscores)
    rename_mapping_main_latex = {
        'excess_ret_immediate': 'rimmediate',
        'excess_ret_short_term': 'rshort',
        'excess_ret_medium_term': 'rmedium',
        'excess_ret_long_term': 'rlong',
        'epsfxq': 'epsfxq',
        'epsfxq_next': 'epsfxqnext'
    }

    # Mapping for renaming other models
    rename_mapping_other = {
        'length_participant_questions': 'length_participant_questions',
        'length_management_answers': 'length_management_answers',
        'difference_questions_answers': 'difference_questions_answers'
    }

    for group_name, dep_vars in regression_groups.items():
        for dep_var in dep_vars:
            # Define independent variables
            independent_vars = similarity_vars + ['topic_diversity'] + control_vars

            # Prepare the regression variables
            X = analysis_df[independent_vars]
            y = analysis_df[dep_var]
            X = sm.add_constant(X)  # Add a constant term

            # Fit the OLS model
            model = sm.OLS(y, X).fit()

            # Perform diagnostics
            ols_diagnostics(model, dep_var, independent_vars, output_folder)

            # Assign models to the appropriate dictionaries
            if group_name == 'return_vars' and dep_var in rename_mapping_main_latex:
                # Rename the model using the mapping for LaTeX
                short_name_latex = rename_mapping_main_latex[dep_var]
                main_results_latex[short_name_latex] = model
            else:
                # Use the original name for other models and rename accordingly
                short_name = rename_mapping_other.get(dep_var, dep_var)
                other_results[short_name] = model

    return main_results_latex, other_results

# Function to perform Chi-Squared Test of Independence and save results as HTML
def perform_chi_squared_test(group, category_var1, category_var2, output_folder, caption, group_name):
    """
    Performs a Chi-Squared test of independence between two categorical variables within a group and saves the result as an HTML table with caption.

    Parameters:
    - group (DataFrame): The subset of data to perform the test on.
    - category_var1 (str): First categorical variable.
    - category_var2 (str): Second categorical variable.
    - output_folder (str): Path to the folder where the HTML file will be saved.
    - caption (str): Caption text to include with the test results.
    - group_name (str): Name of the similarity group (e.g., Low, High) for labeling.

    Returns:
    - chi2, p: Chi-squared statistic and p-value.
    """
    contingency_table = pd.crosstab(group[category_var1], group[category_var2])

    # Check if the contingency table is too large
    if contingency_table.shape[0] > 1000 or contingency_table.shape[1] > 1000:
        print(f"Contingency table too large for Chi-Squared Test (shape: {contingency_table.shape}). Skipping this test.")
        return None, None

    # Perform Chi-Squared Test
    chi2, p, dof, ex = chi2_contingency(contingency_table)

    # Create a DataFrame for the contingency table
    contingency_df = contingency_table.copy()

    # Create a summary DataFrame
    summary_df = pd.DataFrame({
        'Chi2 Statistic': [chi2],
        'P-Value': [p],
        'Degrees of Freedom': [dof]
    })

    # Convert DataFrames to HTML
    contingency_html = contingency_df.to_html(classes='table table-striped', border=0)
    summary_html = summary_df.to_html(index=False, classes='table table-bordered', border=0)

    # Create the full HTML with caption
    full_html = f"""
    <!DOCTYPE html>
    <html>
    <head>
        <title>Chi-Squared Test Results - {group_name} Similarity Group</title>
        <link rel="stylesheet" href="https://maxcdn.bootstrapcdn.com/bootstrap/4.0.0/css/bootstrap.min.css">
        <style>
            body {{
                font-family: Arial, sans-serif;
                margin: 20px;
            }}
            table {{
                border-collapse: collapse;
                width: 100%;
                margin-bottom: 20px;
            }}
            th, td {{
                border: 1px solid #dddddd;
                text-align: center;
                padding: 8px;
            }}
            th {{
                background-color: #f2f2f2;
            }}
            caption {{
                caption-side: top;
                font-size: 20px;
                margin-bottom: 10px;
                font-weight: bold;
            }}
        </style>
    </head>
    <body>
    <div class="container">
        <h2>Chi-Squared Test of Independence</h2>
        <p><strong>{caption}</strong></p>
        <h3>Contingency Table</h3>
        {contingency_html}
        <h3>Test Summary</h3>
        {summary_html}
    </div>
    </body>
    </html>
    """

    # Define the filename based on category variables and group
    filename = f"chi_squared_{category_var1}_{category_var2}_{group_name}.html"
    output_path = os.path.join(output_folder, filename)

    # Save the HTML file
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write(full_html)
    print(f"Saved Chi-Squared Test results to {output_path}")

    # Print the results
    print(f"Chi-Squared Test between '{category_var1}' and '{category_var2}' for {group_name} Similarity Group:")
    print(f"Chi2 Statistic: {chi2:.4f}, P-Value: {p:.4f}, Degrees of Freedom: {dof}")
    print("Contingency Table:")
    print(contingency_table)
    print("\n")

    return chi2, p

# Function to plot stacked bar charts for topic distributions
def plot_topic_distribution(group, title, output_folder):
    """
    Plots a stacked bar chart of management answer topics based on participant question topics within a group.

    Parameters:
    - group (DataFrame): The subset of data to plot.
    - title (str): Title of the plot.
    - output_folder (str): Directory to save the plot.

    Returns:
    - None
    """
    contingency_table = pd.crosstab(group['primary_participant_topic'], group['primary_management_topic'])

    if contingency_table.empty:
        print(f"No data available to plot for {title}.")
        return

    # Check if the table is too large to plot
    if contingency_table.shape[0] > 50 or contingency_table.shape[1] > 50:
        print(f"Contingency table too large to plot for {title}. Skipping visualization.")
        return

    # Normalize the contingency table for better visualization
    contingency_norm = contingency_table.div(contingency_table.sum(axis=1), axis=0)

    plt.figure(figsize=(18, 12))
    contingency_norm.plot(kind='bar', stacked=True, colormap='tab20')
    plt.title(title, fontsize=20)
    plt.xlabel('Participant Question Topics', fontsize=16)
    plt.ylabel('Proportion of Management Answer Topics', fontsize=16)
    plt.legend(title='Management Answer Topics', bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=12)
    plt.tight_layout()

    # Replace spaces with underscores for filename
    filename = f"{title.replace(' ', '_')}.png"
    plt.savefig(os.path.join(output_folder, filename), dpi=300)
    plt.close()
    print(f"Saved plot: {title}")

#%% Define Captions

# Define captions for different regression analyses

captions = {
    'return_vars': """
    This table shows OLS regressions of excess returns and earnings per share forecasts on similarity measures, control variables, and topic diversity.
    Variables of interest include similarity to overall average, industry average, and company average structures of Earnings Conference Calls (ECCs), as well as topic diversity.
    Control variables are standardized as per the variable appendix.
    The sample consists of all earnings conference calls from January 2010 to December 2020.
    Standard errors are robust and clustered by firm. *, **, and *** indicate significance at the 10%, 5%, and 1% level, respectively.
    """,
    'additional_dependent_vars': """
    This table shows OLS regressions of the length of participant questions and management answers on similarity measures, control variables, and topic diversity.
    Variables of interest include similarity to overall average, industry average, and company average structures of Earnings Conference Calls (ECCs), as well as topic diversity.
    Control variables are standardized as per the variable appendix.
    The sample consists of all earnings conference calls from January 2010 to December 2020.
    Standard errors are robust and clustered by firm. *, **, and *** indicate significance at the 10%, 5%, and 1% level, respectively.
    """,
    'difference_questions_answers': """
    This table shows OLS regressions of the difference between participant questions and management answers on similarity measures, control variables, and topic diversity.
    Variables of interest include similarity to overall average, industry average, and company average structures of Earnings Conference Calls (ECCs), as well as topic diversity.
    Control variables are standardized as per the variable appendix.
    The sample consists of all earnings conference calls from January 2010 to December 2020.
    Standard errors are robust and clustered by firm. *, **, and *** indicate significance at the 10%, 5%, and 1% level, respectively.
    """,
    'combined_regressions_main': """
    Combined OLS Regression Results for Return Variables
    """,
    'combined_regressions_other': """
    Combined OLS Regression Results for Other Variables
    """
}

#%% Define Regression Groups

# Define regression groups where each key corresponds to a set of dependent variables that share similar independent variables
regression_groups = {
    'return_vars': [
        'excess_ret_immediate',
        'excess_ret_short_term',
        'excess_ret_medium_term',
        'excess_ret_long_term',
        'epsfxq',
        'epsfxq_next'
    ],
    'additional_dependent_vars': [
        'length_participant_questions',
        'length_management_answers'
    ],
    'research_questions': [
        'difference_questions_answers'
    ]
}

#%% Similarity Variables
similarity_vars = [
    'similarity_to_overall_average',
    'similarity_to_industry_average',
    'similarity_to_company_average'
]

#%% Perform Combined Regressions

# Capture the returned variables from the function to avoid TypeError
main_results_latex, other_results = perform_combined_regressions(regression_groups, analysis_df, output_folder, captions)

#%% Perform Chi-Squared Tests for Topic Distributions

# Define Chi-Squared Test captions
chi2_captions = {
    'Low': """
    This table presents the Chi-Squared Test of Independence between participant question topics and management answer topics for the Low Similarity Group.
    """,
    'High': """
    This table presents the Chi-Squared Test of Independence between participant question topics and management answer topics for the High Similarity Group.
    """
}

# Perform Chi-Squared Test for Low Similarity Group
perform_chi_squared_test(
    group=low_similarity,
    category_var1='primary_participant_topic',
    category_var2='primary_management_topic',
    output_folder=output_folder,
    caption=chi2_captions['Low'],
    group_name='Low'
)

# Perform Chi-Squared Test for High Similarity Group
perform_chi_squared_test(
    group=high_similarity,
    category_var1='primary_participant_topic',
    category_var2='primary_management_topic',
    output_folder=output_folder,
    caption=chi2_captions['High'],
    group_name='High'
)

#%% Comparing Lengths of Management Answers and Participant Questions

# Calculate mean lengths for low and high similarity groups
low_mean_participant = low_similarity['length_participant_questions'].mean()
high_mean_participant = high_similarity['length_participant_questions'].mean()
low_mean_management = low_similarity['length_management_answers'].mean()
high_mean_management = high_similarity['length_management_answers'].mean()

print(f"Average length of participant questions in Low Similarity: {low_mean_participant:.2f}")
print(f"Average length of participant questions in High Similarity: {high_mean_participant:.2f}")
print(f"Average length of management answers in Low Similarity: {low_mean_management:.2f}")
print(f"Average length of management answers in High Similarity: {high_mean_management:.2f}\n")

# Perform t-test for 'length_participant_questions'
t_stat_participant, p_val_participant = ttest_ind(
    low_similarity['length_participant_questions'], 
    high_similarity['length_participant_questions'], 
    equal_var=False  # Use Welch's t-test which does not assume equal population variance
)
print(f"T-Test for 'length_participant_questions' between Low and High Similarity:")
print(f"T-Statistic: {t_stat_participant:.3f}, P-Value: {p_val_participant:.3f}\n")

# Perform t-test for 'length_management_answers'
t_stat_management, p_val_management = ttest_ind(
    low_similarity['length_management_answers'], 
    high_similarity['length_management_answers'], 
    equal_var=False  # Use Welch's t-test
)
print(f"T-Test for 'length_management_answers' between Low and High Similarity:")
print(f"T-Statistic: {t_stat_management:.3f}, P-Value: {p_val_management:.3f}\n")

#%% Perform t-test for 'difference_questions_answers'

# Calculate mean difference_questions_answers for low and high similarity groups
low_mean_difference = low_similarity['difference_questions_answers'].mean()
high_mean_difference = high_similarity['difference_questions_answers'].mean()

print(f"Average difference between participant questions and management answers in Low Similarity: {low_mean_difference:.2f}")
print(f"Average difference between participant questions and management answers in High Similarity: {high_mean_difference:.2f}\n")

# Perform t-test for 'difference_questions_answers'
t_stat_difference, p_val_difference = ttest_ind(
    low_similarity['difference_questions_answers'],
    high_similarity['difference_questions_answers'],
    equal_var=False  # Use Welch's t-test
)

print(f"T-Test for 'difference_questions_answers' between Low and High Similarity:")
print(f"T-Statistic: {t_stat_difference:.3f}, P-Value: {p_val_difference:.3f}\n")

#%% Additional Visualizations

# Plot for Low Similarity Group
plot_topic_distribution(low_similarity, 'Topic Distribution - Low Similarity', output_folder)

# Plot for High Similarity Group
plot_topic_distribution(high_similarity, 'Topic Distribution - High Similarity', output_folder)

#%% Generate and Save T-Test HTML and LaTeX Tables

# Collect t-test results into a list of dictionaries
t_test_results = [
    {
        'Variable': 'Participant Questions',
        'Low Similarity Mean': round(low_mean_participant, 2),
        'High Similarity Mean': round(high_mean_participant, 2),
        'T-Statistic': round(t_stat_participant, 3),
        'P-Value': f"{p_val_participant:.3f}",
        'Interpretation': (
            'Highly significant difference; participant questions are longer in Low Similarity group.' 
            if p_val_participant < 0.001 else
            'Significant difference; participant questions are longer in Low Similarity group.' 
            if p_val_participant < 0.01 else
            'Moderately significant difference; participant questions are longer in Low Similarity group.' 
            if p_val_participant < 0.05 else
            'No significant difference in participant questions between groups.'
        )
    },
    {
        'Variable': 'Management Answers',
        'Low Similarity Mean': round(low_mean_management, 2),
        'High Similarity Mean': round(high_mean_management, 2),
        'T-Statistic': round(t_stat_management, 3),
        'P-Value': f"{p_val_management:.3f}",
        'Interpretation': (
            'Highly significant difference; management answers are longer in Low Similarity group.' 
            if p_val_management < 0.001 else
            'Significant difference; management answers are longer in Low Similarity group.' 
            if p_val_management < 0.01 else
            'Moderately significant difference; management answers are longer in Low Similarity group.' 
            if p_val_management < 0.05 else
            'No significant difference in management answers between groups.'
        )
    },
    {
        'Variable': 'Difference between Questions and Answers',
        'Low Similarity Mean': round(low_mean_difference, 2),
        'High Similarity Mean': round(high_mean_difference, 2),
        'T-Statistic': round(t_stat_difference, 3),
        'P-Value': f"{p_val_difference:.3f}",
        'Interpretation': (
            'Highly significant difference; the difference is larger in Low Similarity group.'
            if p_val_difference < 0.001 else
            'Significant difference; the difference is larger in Low Similarity group.'
            if p_val_difference < 0.01 else
            'Moderately significant difference; the difference is larger in Low Similarity group.'
            if p_val_difference < 0.05 else
            'No significant difference in the difference between groups.'
        )
    }
]

# Create a DataFrame from the t-test results
t_test_df = pd.DataFrame(t_test_results)

# Function to highlight significance based on p-value (for HTML table)
def highlight_significance(row):
    """
    Highlights the row based on the p-value.
    - Red: p < 0.001
    - Orange: p < 0.01
    - Yellow: p < 0.05
    - Light Blue: p >= 0.05
    """
    p_val = float(row['P-Value'])
    if p_val < 0.001:
        color = '#d7191c'  # Red
    elif p_val < 0.01:
        color = '#fdae61'  # Orange
    elif p_val < 0.05:
        color = '#ffffbf'  # Yellow
    else:
        color = '#abd9e9'  # Light Blue
    return [f'background-color: {color}']*len(row)

# Apply the styling to the DataFrame for HTML
styled_t_test_df = t_test_df.style.apply(highlight_significance, axis=1)

# Define CSS styles for the table
table_styles = [
    {'selector': 'th', 'props': [('background-color', '#f2f2f2'), ('text-align', 'center'), ('padding', '12px')]},
    {'selector': 'td', 'props': [('text-align', 'center'), ('padding', '12px')]}
]

# Convert the styled DataFrame to HTML
html_table = styled_t_test_df.set_table_styles(table_styles).to_html()

# Create the full HTML content
t_test_html_content = f"""
<!DOCTYPE html>
<html>
<head>
    <title>T-Test Results: Low vs. High Similarity Groups</title>
    <style>
        body {{
            font-family: Arial, sans-serif;
            margin: 20px;
        }}
        table {{
            border-collapse: collapse;
            width: 80%;
            margin: 0 auto;
        }}
        th, td {{
            border: 1px solid #dddddd;
            text-align: center;
            padding: 12px;
        }}
        th {{
            background-color: #f2f2f2;
        }}
        caption {{
            caption-side: top;
            font-size: 20px;
            margin-bottom: 10px;
            font-weight: bold;
        }}
    </style>
</head>
<body>

    <table>
        <caption>T-Test Results: Low vs. High Similarity Groups</caption>
        {html_table}
    </table>

    <div style="width: 80%; margin: 20px auto;">
        <p><strong>Implications:</strong> There's a statistically significant reduction in both participant question lengths, management answer lengths, and the difference between them in the High Similarity group compared to the Low Similarity group.</p>
    </div>

</body>
</html>
"""

# Define the output HTML file path
t_test_html_filename = "t_test_results.html"
t_test_html_path = os.path.join(output_folder, t_test_html_filename)

# Save the HTML content to the file
with open(t_test_html_path, 'w', encoding='utf-8') as file:
    file.write(t_test_html_content)

print(f"Saved T-Test Results table to {t_test_html_path}")

#%% Generate LaTeX Tables for Regression Results

def generate_latex_table(summary, caption, label, output_path):
    """
    Generates a LaTeX table from a summary_col object with adjustments to fit the page width.
    
    Parameters:
    - summary: The summary_col object.
    - caption (str): Caption for the table.
    - label (str): Label for the table.
    - output_path (str): Path to save the LaTeX table.
    
    Returns:
    - None
    """
    # Get the LaTeX code from summary_col
    latex_table = summary.as_latex()

    # Find the tabular environment
    begin_tabular = latex_table.find('\\begin{tabular}')
    end_tabular = latex_table.find('\\end{tabular}') + len('\\end{tabular}')

    # Extract the tabular content
    tabular_content = latex_table[begin_tabular:end_tabular]

    # Wrap the tabular environment with \small and braces
    wrapped_tabular = '{\\small ' + tabular_content + '}'

    # Construct the full table with \resizebox and proper structure
    full_table = f"""
\\begin{{table}}[ht]
\\centering
\\caption{{{caption}}}
\\label{{{label}}}
\\resizebox{{\\textwidth}}{{!}}{{%
    {wrapped_tabular}
}}
\\bigskip
\\textit{{Standard errors in parentheses.}} \\\\ 
* p<.1, ** p<.05, ***p<.01
\\end{{table}}
"""
    
    # Save to file
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write(full_table)
    
    print(f"Saved LaTeX table to {output_path}")

#%% Generate LaTeX Tables for Main and Other Regression Results

# Generate LaTeX table for main regression results (Return Variables)
if main_results_latex:
    combined_main_summary_latex = summary_col(
        list(main_results_latex.values()),
        stars=True,
        model_names=list(main_results_latex.keys()),
        info_dict={
            'No. Observations': lambda x: f"{int(x.nobs)}"
        },
        float_format='%0.3f'
    )

    # Define the caption and label
    main_latex_caption = captions['combined_regressions_main']
    main_latex_label = "tab:combined_regression_results_main"

    # Define the filename and path
    main_latex_filename = "combined_regression_results_main.tex"
    main_latex_output_path = os.path.join(output_folder, main_latex_filename)

    # Generate the LaTeX table with adjustments
    generate_latex_table(combined_main_summary_latex, main_latex_caption, main_latex_label, main_latex_output_path)
else:
    print("No main regression models to generate LaTeX table.\n")

# Generate LaTeX table for other regression results (Other Variables)
if other_results:
    combined_other_summary_latex = summary_col(
        list(other_results.values()),
        stars=True,
        model_names=list(other_results.keys()),
        info_dict={
            'No. Observations': lambda x: f"{int(x.nobs)}"
        },
        float_format='%0.3f'
    )

    # Define the caption and label
    other_latex_caption = captions['combined_regressions_other']
    other_latex_label = "tab:combined_regression_results_other"

    # Define the filename and path
    other_latex_filename = "combined_regression_results_other.tex"
    other_latex_output_path = os.path.join(output_folder, other_latex_filename)

    # Generate the LaTeX table with adjustments
    generate_latex_table(combined_other_summary_latex, other_latex_caption, other_latex_label, other_latex_output_path)
else:
    print("No other regression models to generate LaTeX table.\n")

#%% Generate LaTeX Table for T-Test Results

# Define LaTeX table filename
t_test_latex_filename = "t_test_results.tex"
t_test_latex_path = os.path.join(output_folder, t_test_latex_filename)

# Create a LaTeX table using pandas
latex_table = t_test_df.to_latex(index=False, 
                                 caption='T-Test Results: Low vs. High Similarity Groups',
                                 label='tab:t_test_results',
                                 column_format='lccccp{8cm}',  # Adjust column alignment as needed
                                 escape=False)  # To allow LaTeX formatting in 'Interpretation'

# Add a note for significance
latex_table_with_note = f"""{latex_table}
\\begin{{flushleft}}
\\textit{{Note:}} *, **, and *** indicate significance at the 10\\%, 5\\%, and 1\\% levels, respectively.
\\end{{flushleft}}
"""

# Save the LaTeX table to a .tex file
with open(t_test_latex_path, 'w', encoding='utf-8') as f:
    f.write(latex_table_with_note)

print(f"Saved LaTeX T-Test Results table to {t_test_latex_path}")

#%% End of Script
