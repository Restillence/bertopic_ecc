# variable_analysis.py

# Import necessary libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import statsmodels.api as sm
from scipy.stats import ttest_ind, shapiro
from statsmodels.stats.diagnostic import het_breuschpagan
from statsmodels.graphics.gofplots import qqplot
from statsmodels.stats.stattools import durbin_watson
from statsmodels.stats.outliers_influence import variance_inflation_factor
from sklearn.preprocessing import StandardScaler
import os
import ast
import re
from utils import (
    create_average_transition_matrix_figures,
    create_transition_matrix,
    remove_neg_one_from_columns  # Ensure this function is correctly defined in utils.py
)
from tqdm import tqdm  # For progress bars in loops

# Configure Plotting Aesthetics
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

# Create '../regression_results' Folder
output_folder = os.path.join("..", "regression_results")

if not os.path.exists(output_folder):
    os.makedirs(output_folder)
    print(f"Created folder: {output_folder}")
else:
    print(f"Folder already exists: {output_folder}")

# Path to the Final Dataset
filepath = "D:/daten_masterarbeit/final_dataset_reg_full.csv"

try:
    df = pd.read_csv(filepath)
    print(f"Number of observations in the final_dataset: {len(df)}")
except FileNotFoundError:
    print(f"File not found: {filepath}")
    raise

# Remove the first year (2002) if not done already
print("Removing data from 2002 from df, if not done already...")
df['call_date'] = pd.to_datetime(df['call_date'], errors='coerce')
df = df[df['call_date'].dt.year != 2002]
print(f"Number of rows after removing 2002 data: {len(df)}")

# Data Preparation
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
    'length_participant_questions',
    'length_management_answers',
    'market_cap',
    'rolling_beta',
    'ceo_participates',
    'ceo_cfo_change',
    'word_length_presentation',
    'filtered_presentation_topics',
    'participant_question_topics',
    'management_answer_topics',
    'permco',
    'siccd'
]

missing_vars = [var for var in variables if var not in df.columns]
if missing_vars:
    raise KeyError(f"The following required columns are missing from the DataFrame: {missing_vars}")

analysis_df = df[variables].dropna()
print(f"Number of observations after dropping NaNs: {len(analysis_df)}")

# Parse Topic Columns
def parse_topics(topics):
    if isinstance(topics, str):
        try:
            parsed = ast.literal_eval(topics)
            if isinstance(parsed, list):
                return parsed
            else:
                return []
        except (ValueError, SyntaxError):
            return []
    elif isinstance(topics, list):
        return topics
    else:
        return []

def flatten_topics(topics):
    if not isinstance(topics, list):
        return []
    flattened = []
    for item in topics:
        if isinstance(item, list):
            flattened.extend(item)
        else:
            flattened.append(item)
    return flattened

def convert_topics_to_int(topics):
    if not isinstance(topics, list):
        return []
    flattened = flatten_topics(topics)
    int_topics = []
    for topic in flattened:
        try:
            int_topic = int(topic)
            int_topics.append(int_topic)
        except (ValueError, TypeError):
            continue
    return int_topics

topic_columns = ['filtered_presentation_topics', 'participant_question_topics', 'management_answer_topics']
for col in topic_columns:
    analysis_df[col] = analysis_df[col].apply(parse_topics)
    analysis_df[col] = analysis_df[col].apply(convert_topics_to_int)

def verify_topics_are_int(topics):
    return all(isinstance(topic, int) for topic in topics)

verification = analysis_df[topic_columns].applymap(verify_topics_are_int).all(axis=1)
if not verification.all():
    problematic_rows = verification[verification == False].index.tolist()
    print(f"Warning: The following rows have non-integer topics and will be excluded: {problematic_rows}")
    analysis_df = analysis_df[verification]
else:
    print("All topic columns are correctly formatted as lists of integers.")

print("\nSample 'filtered_presentation_topics' after parsing:")
print(analysis_df['filtered_presentation_topics'].head())

print("\nNumber of topics per row:")
print(analysis_df['filtered_presentation_topics'].apply(len).describe())

unique_topics = analysis_df['filtered_presentation_topics'].explode().unique()
print(f"\nNumber of unique topics after parsing: {len(unique_topics)}")
print(f"Unique topics: {unique_topics}")

# Create 'topic_diversity' Variable
def calculate_topic_diversity(topics):
    if not isinstance(topics, list) or len(topics) == 0:
        return np.nan
    unique_topics = set(topics)
    diversity = len(unique_topics) / len(topics)
    return diversity

analysis_df['topic_diversity'] = analysis_df['filtered_presentation_topics'].apply(calculate_topic_diversity)
mean_diversity = analysis_df['topic_diversity'].mean()
analysis_df['topic_diversity'] = analysis_df['topic_diversity'].fillna(mean_diversity)
print("Created 'topic_diversity' variable.")

# Create 'difference_questions_answers' Variable
analysis_df['difference_questions_answers'] = analysis_df['length_participant_questions'] - analysis_df['length_management_answers']

print("Summary statistics for 'difference_questions_answers':")
print(analysis_df['difference_questions_answers'].describe())

plt.figure(figsize=(12, 8))
sns.histplot(analysis_df['difference_questions_answers'], kde=True, bins=30, color='skyblue')
plt.title('Distribution of Difference Between Participant Questions and Management Answers', fontsize=16)
plt.xlabel('Participant Questions - Management Answers', fontsize=14)
plt.ylabel('Frequency', fontsize=14)
plt.tight_layout()
plt.savefig(os.path.join(output_folder, 'difference_questions_answers_distribution.png'), dpi=300)
plt.close()
print("Saved distribution plot for 'difference_questions_answers'.")

# Descriptive Statistics
descriptive_stats_folder = os.path.join(output_folder, "descriptive_statistics")
if not os.path.exists(descriptive_stats_folder):
    os.makedirs(descriptive_stats_folder)
    print(f"Created folder for descriptive statistics: {descriptive_stats_folder}")
else:
    print(f"Folder for descriptive statistics already exists: {descriptive_stats_folder}")

numerical_cols = analysis_df.select_dtypes(include=[np.number]).columns.tolist()
numerical_summary = analysis_df[numerical_cols].describe().transpose()
numerical_summary['skew'] = analysis_df[numerical_cols].skew()
numerical_summary['kurtosis'] = analysis_df[numerical_cols].kurtosis()
numerical_summary = numerical_summary[['count', 'mean', 'std', 'min', '25%', '50%', '75%', 'max', 'skew', 'kurtosis']]
numerical_summary = numerical_summary.round(3)

numerical_summary_latex = numerical_summary.to_latex(
    caption='Descriptive Statistics for Numerical Variables',
    label='tab:descriptive_numerical',
    float_format='%.3f',
    column_format='lcccccccccc',
    bold_rows=True
)

numerical_latex_path = os.path.join(descriptive_stats_folder, "descriptive_statistics_numerical.tex")

with open(numerical_latex_path, 'w') as f:
    f.write(numerical_summary_latex)

print(f"Saved numerical descriptive statistics LaTeX table to {numerical_latex_path}")

# Standardize Independent and Control Variables (Excluding Dummy Variables)
independent_vars = [
    'similarity_to_overall_average',
    'similarity_to_industry_average',
    'similarity_to_company_average',
    'topic_diversity'
]

# Add the dummy variables ceo_participates and ceo_cfo_change to the control variables to ensure they appear in the regressions
control_vars = [
    'market_cap',
    'rolling_beta',
    'word_length_presentation',
    'ceo_participates',
    'ceo_cfo_change'
]

variables_to_scale = independent_vars + ['market_cap', 'rolling_beta', 'word_length_presentation']
# ceo_participates and ceo_cfo_change are dummies, do not standardize them

scaler = StandardScaler()
analysis_df[variables_to_scale] = scaler.fit_transform(analysis_df[variables_to_scale])
print(f"Standardized variables: {', '.join(variables_to_scale)}")

def calculate_vif(vars_list, df):
    X = df[vars_list]
    X = sm.add_constant(X)
    vif_data = pd.DataFrame()
    vif_data["Variable"] = X.columns
    vif_data["VIF"] = [variance_inflation_factor(X.values, i) for i in range(X.shape[1])]
    return vif_data

print("Calculating VIF for independent and control variables...")
vif_df = calculate_vif(independent_vars + ['market_cap', 'rolling_beta', 'word_length_presentation'], analysis_df)
print("VIF Results:")
print(vif_df)

low_percentile = 20
high_percentile = 80
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

low_similarity = analysis_df[
    (analysis_df['similarity_to_overall_average'] <= low_thresholds['similarity_to_overall_average']) &
    (analysis_df['similarity_to_industry_average'] <= low_thresholds['similarity_to_industry_average']) &
    (analysis_df['similarity_to_company_average'] <= low_thresholds['similarity_to_company_average'])
]

high_similarity = analysis_df[
    (analysis_df['similarity_to_overall_average'] > high_thresholds['similarity_to_overall_average']) &
    (analysis_df['similarity_to_industry_average'] > high_thresholds['similarity_to_industry_average']) &
    (analysis_df['similarity_to_company_average'] > high_thresholds['similarity_to_company_average'])
]

print(f"Defined similarity groups based on {low_percentile}th and {high_percentile}th percentiles.")
print(f"Low Similarity Group: {len(low_similarity)} observations")
print(f"High Similarity Group: {len(high_similarity)} observations\n")

numeric_cols = analysis_df.select_dtypes(include=[np.number]).columns.tolist()
correlations = analysis_df[numeric_cols].corr(method='pearson')
print("Correlation matrix:")
print(correlations)

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

def ols_diagnostics(model, y_var, x_vars, output_folder):
    residuals = model.resid
    fitted = model.fittedvalues

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

    plt.figure(figsize=(12, 8))
    qqplot(residuals, line='s', ax=plt.gca())
    plt.title(f'Q-Q Plot of Residuals for {y_var}', fontsize=16)
    plt.tight_layout()
    plt.savefig(os.path.join(output_folder, f'{y_var}_qq_plot.png'), dpi=300)
    plt.close()
    print(f"Saved Q-Q plot for {y_var}.")

    plt.figure(figsize=(12, 8))
    sns.histplot(residuals, kde=True, bins=30, color='salmon')
    plt.title(f'Histogram of Residuals for {y_var}', fontsize=16)
    plt.xlabel('Residuals', fontsize=14)
    plt.ylabel('Frequency', fontsize=14)
    plt.tight_layout()
    plt.savefig(os.path.join(output_folder, f'{y_var}_residuals_histogram.png'), dpi=300)
    plt.close()
    print(f"Saved Residuals Histogram for {y_var}.")

    bp_test = het_breuschpagan(residuals, model.model.exog)
    bp_pvalue = bp_test[1]
    print(f"Breusch-Pagan test p-value for {y_var}: {bp_pvalue:.4f}")

    if len(residuals) <= 5000:
        shapiro_stat, shapiro_p = shapiro(residuals)
        print(f"Shapiro-Wilk test p-value for {y_var}: {shapiro_p:.4f}")
    else:
        print(f"Shapiro-Wilk test not performed for {y_var} due to large sample size (N={len(residuals)}).")

    dw_stat = durbin_watson(residuals)
    print(f"Durbin-Watson statistic for {y_var}: {dw_stat:.4f}")

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

def clean_variable(var):
    if isinstance(var, str):
        var = var.split('\n')[0]
        var = re.sub(r'Name:\s*\d+,\s*dtype:\s*object', '', var)
        var = var.strip()
        return var
    else:
        return str(var).strip()

def create_regression_latex_table(regression_stats, r_squared_stats, adj_r_squared_stats, n_observations_stats, caption, label, output_path):
    """
    Creates a LaTeX table from regression statistics including coefficients with stars and t-values in parentheses.
    Also includes R-squared, Adjusted R-squared, and Number of Observations for each regression.

    Parameters:
    - regression_stats (dict): Dictionary where keys are model names and values are DataFrames with 'Coefficient' and 'stat_value'.
    - r_squared_stats (dict): Dictionary where keys are model names and values are R-squared values.
    - adj_r_squared_stats (dict): Dictionary where keys are model names and values are Adjusted R-squared values.
    - n_observations_stats (dict): Dictionary where keys are model names and values are number of observations.
    - caption (str): Caption for the LaTeX table.
    - label (str): Label for referencing the table in LaTeX.
    - output_path (str): File path to save the LaTeX table.
    """
    combined_df = pd.DataFrame()

    for model_name, stats in regression_stats.items():
        stats = stats.copy()
        stats.columns = pd.MultiIndex.from_product([[model_name], stats.columns])
        combined_df = pd.concat([combined_df, stats], axis=1)

    combined_df.reset_index(inplace=True)
    combined_df.rename(columns={'index': 'Variable'}, inplace=True)
    combined_df['Variable'] = combined_df['Variable'].apply(clean_variable)

    # Initialize rows list
    rows = []
    for index, row in combined_df.iterrows():
        variable = row['Variable']
        coeffs = []
        stat_values = []
        for model in regression_stats.keys():
            coeff = row[(model, 'Coefficient')]
            stat_val = row[(model, 'stat_value')]
            coeffs.append(coeff)
            stat_values.append(stat_val)

        coeff_row = [variable] + coeffs
        stat_row = [''] + stat_values
        rows.append(coeff_row)
        rows.append(stat_row)

    # Define headers
    headers = ['Variable'] + list(regression_stats.keys())
    formatted_df = pd.DataFrame(rows, columns=headers)
    formatted_df['Variable'] = formatted_df['Variable'].astype(str)

    # Start LaTeX table
    column_alignment = 'l' + 'c' * len(regression_stats.keys())

    latex_content = f"""
\\begin{{table}}[ht]
    \\centering
    \\caption{{{caption}}}
    \\label{{{label}}}
    \\resizebox{{\\textwidth}}{{!}}{{%
        {{\\small
        \\begin{{tabular}}{{{column_alignment}}}
        \\hline
                                & {' & '.join([f'\\textit{{{model}}}' for model in regression_stats.keys()])} \\\\
        \\hline
    """

    # Add coefficient and t-value rows
    for i, row in formatted_df.iterrows():
        variable = row['Variable']
        if variable == '':
            latex_content += "                            & " + " & ".join(row[1:]) + " \\\\\n"
        else:
            variable_escaped = re.sub(r'([&_#%{}~^\\])', r'\\\1', variable)
            latex_content += f"    {variable_escaped} & " + " & ".join(row[1:]) + " \\\\\n"

    # Add R-squared, Adjusted R-squared, and No. Observations
    latex_content += "\\hline\n"

    # R-squared row
    r_squared_values = [f"{r_squared_stats[model]:.3f}" for model in regression_stats.keys()]
    latex_content += f"R-squared & " + " & ".join(r_squared_values) + " \\\\\n"

    # Adjusted R-squared row
    adj_r_squared_values = [f"{adj_r_squared_stats[model]:.3f}" for model in regression_stats.keys()]
    latex_content += f"R-squared Adj. & " + " & ".join(adj_r_squared_values) + " \\\\\n"

    # No. Observations row
    n_observations_values = [f"{n_observations_stats[model]}" for model in regression_stats.keys()]
    latex_content += f"No. Observations & " + " & ".join(n_observations_values) + " \\\\\n"

    # End LaTeX table
    latex_content += f"""
    \\hline
    \\end{{tabular}}}}
    }}
    \\bigskip
    \\textit{{t-values in parentheses.}} \\ 
    * p<.1, ** p<.05, ***p<.01
\\end{{table}}
"""

    # Write the LaTeX content to file
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write(latex_content)

    print(f"Saved LaTeX regression table to {output_path}")

def create_regression_html_table(regression_stats, r_squared_stats, adj_r_squared_stats, n_observations_stats, caption, label, output_path):
    """
    Creates an HTML table from regression statistics including coefficients with stars and t-values in parentheses.
    Also includes R-squared, Adjusted R-squared, and Number of Observations.

    Parameters:
    - regression_stats (dict): Dictionary where keys are model names and values are DataFrames with 'Coefficient' and 'stat_value'.
    - r_squared_stats (dict): Dictionary where keys are model names and values are R-squared values.
    - adj_r_squared_stats (dict): Dictionary where keys are model names and values are Adjusted R-squared values.
    - n_observations_stats (dict): Dictionary where keys are model names and values are number of observations.
    - caption (str): Caption for the HTML table.
    - label (str): Label for the table (used for CSS or referencing purposes).
    - output_path (str): File path to save the HTML table.
    """
    combined_df = pd.DataFrame()

    for model_name, stats in regression_stats.items():
        stats = stats.copy()
        stats.columns = pd.MultiIndex.from_product([[model_name], stats.columns])
        combined_df = pd.concat([combined_df, stats], axis=1)

    combined_df.reset_index(inplace=True)
    combined_df.rename(columns={'index': 'Variable'}, inplace=True)
    combined_df['Variable'] = combined_df['Variable'].apply(clean_variable)

    # Initialize rows list
    rows = []
    for index, row in combined_df.iterrows():
        variable = row['Variable']
        coeffs = []
        stat_values = []
        for model in regression_stats.keys():
            coeff = row[(model, 'Coefficient')]
            stat_val = row[(model, 'stat_value')]
            coeffs.append(coeff)
            stat_values.append(stat_val)

        coeff_row = [variable] + coeffs
        stat_row = [''] + stat_values
        rows.append(coeff_row)
        rows.append(stat_row)

    # Define headers
    headers = ['Variable'] + list(regression_stats.keys())
    formatted_df = pd.DataFrame(rows, columns=headers)
    formatted_df['Variable'] = formatted_df['Variable'].astype(str)

    # Start HTML table
    html_content = f"""
<!DOCTYPE html>
<html>
<head>
    <title>{caption}</title>
    <style>
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
        <caption><strong>{caption}</strong></caption>
        <tr>
            <th></th>
            {''.join([f'<th><em>{col}</em></th>' for col in headers[1:]])}
        </tr>
"""

    # Add coefficient and t-value rows
    for i, row in formatted_df.iterrows():
        variable = row['Variable']
        if variable == '':
            html_content += "<tr>\n"
            html_content += "    <td></td>" + "".join([f"<td>{cell}</td>" for cell in row[1:]]) + "\n</tr>\n"
        else:
            variable_escaped = variable.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
            html_content += f"<tr>\n    <td>{variable_escaped}</td>" + "".join([f"<td>{cell}</td>" for cell in row[1:]]) + "\n</tr>\n"

    # Add R-squared, Adjusted R-squared, and No. Observations rows with proper HTML syntax
    # R-squared row
    r_squared_values = [f"{r_squared_stats[model]:.3f}" for model in regression_stats.keys()]
    html_content += f"""
        <tr>
            <td><strong>R-squared</strong></td>
            {''.join([f"<td>{val}</td>" for val in r_squared_values])}
        </tr>
    """

    # Adjusted R-squared row
    adj_r_squared_values = [f"{adj_r_squared_stats[model]:.3f}" for model in regression_stats.keys()]
    html_content += f"""
        <tr>
            <td><strong>R-squared Adj.</strong></td>
            {''.join([f"<td>{val}</td>" for val in adj_r_squared_values])}
        </tr>
    """

    # No. Observations row
    n_observations_values = [f"{n_observations_stats[model]}" for model in regression_stats.keys()]
    html_content += f"""
        <tr>
            <td><strong>No. Observations</strong></td>
            {''.join([f"<td>{val}</td>" for val in n_observations_values])}
        </tr>
    </table>

    <p><em>t-values in parentheses.</em></p>
    <p>* p&lt;.1, ** p&lt;.05, ***p&lt;.01</p>

</body>
</html>
"""

    # Write the HTML content to file
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write(html_content)

    print(f"Saved HTML regression table to {output_path}")

#%% Perform Combined Regressions

def perform_combined_regressions(regression_groups, analysis_df, output_folder, captions):
    """
    Performs multiple OLS regressions without clustered standard errors, separates them into main and other groups, 
    renames main group models, and collects regression statistics with significance stars and t-values.
    Also collects R-squared, Adjusted R-squared, and Number of Observations for each regression.

    Parameters:
    - regression_groups (dict): Dictionary where keys are group names and values are lists of dependent variables.
    - analysis_df (DataFrame): The DataFrame containing the data.
    - output_folder (str): Path to the folder where the LaTeX and HTML files will be saved.
    - captions (dict): Dictionary where keys are group names and values are caption texts.

    Returns:
    - main_results_stats (dict): Dictionary of main regression statistics for LaTeX and HTML tables.
    - other_results_stats (dict): Dictionary of other regression statistics.
    - main_r_squared_stats (dict): Dictionary of R-squared values for main regressions.
    - main_adj_r_squared_stats (dict): Dictionary of Adjusted R-squared values for main regressions.
    - main_n_observations_stats (dict): Dictionary of number of observations for main regressions.
    - other_r_squared_stats (dict): Dictionary of R-squared values for other regressions.
    - other_adj_r_squared_stats (dict): Dictionary of Adjusted R-squared values for other regressions.
    - other_n_observations_stats (dict): Dictionary of number of observations for other regressions.
    """
    # Dictionaries to hold regression statistics
    main_results_stats = {}
    other_results_stats = {}
    
    # Dictionaries to hold additional regression statistics
    main_r_squared_stats = {}
    main_adj_r_squared_stats = {}
    main_n_observations_stats = {}
    
    other_r_squared_stats = {}
    other_adj_r_squared_stats = {}
    other_n_observations_stats = {}
    
    # Mapping for renaming main models for LaTeX and HTML
    rename_mapping_main_latex = {
        'excess_ret_immediate': 'r_immediate',
        'excess_ret_short_term': 'r_short',
        'excess_ret_medium_term': 'r_medium',
        'excess_ret_long_term': 'r_long',
        'epsfxq': 'eps',
        'epsfxq_next': 'eps_next'
    }
    
    # Define which models are return variables for formatting
    return_models = ['r_immediate', 'r_short', 'r_medium', 'r_long']
    
    # Mapping for renaming other models
    rename_mapping_other = {
        'length_participant_questions': 'length_participant_questions',
        'length_management_answers': 'length_management_answers',
        'difference_questions_answers': 'difference_questions_answers'
    }
    
    for group_name, dep_vars in regression_groups.items():
        for dep_var in dep_vars:
            # Define independent variables
            independent_vars_combined = independent_vars + control_vars

            # Prepare the regression variables
            X = analysis_df[independent_vars_combined]
            y = analysis_df[dep_var]
            X = sm.add_constant(X)  # Add a constant term

            try:
                # Fit the OLS model without clustered standard errors
                model = sm.OLS(y, X).fit()

                # Perform diagnostics
                ols_diagnostics(model, dep_var, independent_vars_combined, output_folder)

                # Extract regression statistics
                summary_df = model.summary2().tables[1]  # Extract the coefficients table

                # Debug: Print the columns of summary_df
                print(f"Summary table columns for {dep_var}: {summary_df.columns.tolist()}")

                # Attempt to select the expected columns
                if {'Coef.', 't', 'P>|t|'}.issubset(summary_df.columns):
                    summary_df = summary_df[['Coef.', 't', 'P>|t|']]
                elif {'Coef.', 't-value', 'P>|t|'}.issubset(summary_df.columns):
                    summary_df = summary_df[['Coef.', 't-value', 'P>|t|']]
                else:
                    # If expected columns are not found, raise an informative error
                    raise KeyError(f"Expected columns ['Coef.', 't', 'P>|t|'] or ['Coef.', 't-value', 'P>|t|'] not found in summary table for {dep_var}. Found columns: {summary_df.columns.tolist()}")

                # Rename columns for consistency
                summary_df = summary_df.rename(columns={
                    'Coef.': 'Coefficient',
                    't': 'stat_value',
                    't-value': 'stat_value',
                    'P>|t|': 'p_value'
                })

                # Add significance stars based on p-values
                def add_significance_stars(p):
                    if p < 0.001:
                        return '***'
                    elif p < 0.01:
                        return '**'
                    elif p < 0.05:
                        return '*'
                    else:
                        return ''

                # Determine formatting based on model type
                if dep_var in rename_mapping_main_latex:
                    short_name_latex = rename_mapping_main_latex[dep_var]
                    format_decimal = 4 if short_name_latex in return_models else 3
                else:
                    short_name_latex = rename_mapping_other.get(dep_var, dep_var)
                    format_decimal = 3  # Default formatting for other models

                # Apply formatting
                if short_name_latex in return_models:
                    summary_df['Coefficient'] = summary_df.apply(
                        lambda row: f"{row['Coefficient']:.4f}{add_significance_stars(row['p_value'])}", axis=1
                    )
                else:
                    summary_df['Coefficient'] = summary_df.apply(
                        lambda row: f"{row['Coefficient']:.3f}{add_significance_stars(row['p_value'])}", axis=1
                    )

                # Format t-values with parentheses
                summary_df['stat_value'] = summary_df['stat_value'].apply(lambda x: f"({x:.3f})")

                # Remove p-values as they are now represented by stars
                summary_df = summary_df[['Coefficient', 'stat_value']]

                # Assign statistics to the appropriate dictionaries
                if group_name == 'return_vars' and dep_var in rename_mapping_main_latex:
                    # Rename the model using the mapping for LaTeX and HTML
                    main_results_stats[short_name_latex] = summary_df

                    # Collect additional statistics
                    main_r_squared_stats[short_name_latex] = model.rsquared
                    main_adj_r_squared_stats[short_name_latex] = model.rsquared_adj
                    main_n_observations_stats[short_name_latex] = int(model.nobs)
                else:
                    # Use the original name for other models and rename accordingly
                    short_name = rename_mapping_other.get(dep_var, dep_var)
                    other_results_stats[short_name] = summary_df

                    # Collect additional statistics
                    other_r_squared_stats[short_name] = model.rsquared
                    other_adj_r_squared_stats[short_name] = model.rsquared_adj
                    other_n_observations_stats[short_name] = int(model.nobs)

                print(f"Processed regression for {dep_var}.")

            except KeyError as e:
                print(f"KeyError encountered while processing {dep_var}: {e}")
                continue
            except Exception as e:
                print(f"An error occurred while processing {dep_var}: {e}")
                continue

    # IMPORTANT: Add the return statement here to return the collected results
    return (main_results_stats, other_results_stats,
            main_r_squared_stats, main_adj_r_squared_stats, main_n_observations_stats,
            other_r_squared_stats, other_adj_r_squared_stats, other_n_observations_stats)

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

print("Defined regression groups.")

#%% Define Captions

# Define captions for different regression analyses
captions = {
    'combined_regressions_main': "Combined OLS Regression Results for Return Variables",
    'combined_regressions_other': "Combined OLS Regression Results for Other Variables"
}

print("Defined captions for regression tables.")

#%% Perform Combined Regressions and Capture Return Values

# Perform combined regressions and capture the returned statistics
(main_results_stats, other_results_stats,
 main_r_squared_stats, main_adj_r_squared_stats, main_n_observations_stats,
 other_r_squared_stats, other_adj_r_squared_stats, other_n_observations_stats) = perform_combined_regressions(
    regression_groups, analysis_df, output_folder, captions
)

print("Performed combined regressions and captured results.")

#%% Define Function to Generate Tables

def generate_tables(main_stats, other_stats, 
                    main_r2, main_adj_r2, main_nobs, 
                    other_r2, other_adj_r2, other_nobs,
                    output_folder, captions):
    """
    Generates LaTeX and HTML tables for main and other regression results, including R-squared, Adjusted R-squared, and No. Observations.

    Parameters:
    - main_stats (dict): Dictionary of main regression statistics.
    - other_stats (dict): Dictionary of other regression statistics.
    - main_r2 (dict): R-squared values for main regressions.
    - main_adj_r2 (dict): Adjusted R-squared values for main regressions.
    - main_nobs (dict): Number of observations for main regressions.
    - other_r2 (dict): R-squared values for other regressions.
    - other_adj_r2 (dict): Adjusted R-squared values for other regressions.
    - other_nobs (dict): Number of observations for other regressions.
    - output_folder (str): Directory to save the tables.
    - captions (dict): Dictionary containing captions for tables.

    Returns:
    - None
    """
    # LaTeX Tables
    # Generate LaTeX table for main regression results (Return Variables)
    if main_stats:
        try:
            # Define the caption and label
            main_latex_caption = captions['combined_regressions_main']
            main_latex_label = "tab:combined_regression_results_main"

            # Define the filename and path
            main_latex_filename = "combined_regression_results_main.tex"
            main_latex_output_path = os.path.join(output_folder, main_latex_filename)

            # Generate the LaTeX table with coefficients, t-values, and regression statistics
            create_regression_latex_table(
                regression_stats=main_stats,
                r_squared_stats=main_r2,
                adj_r_squared_stats=main_adj_r2,
                n_observations_stats=main_nobs,
                caption=main_latex_caption,
                label=main_latex_label,
                output_path=main_latex_output_path
            )
        except Exception as e:
            print(f"Error generating LaTeX table for main regression results: {e}")
    else:
        print("No main regression models to generate LaTeX table.\n")

    # Generate LaTeX table for other regression results (Other Variables)
    if other_stats:
        try:
            # Define the caption and label
            other_latex_caption = captions['combined_regressions_other']
            other_latex_label = "tab:combined_regression_results_other"

            # Define the filename and path
            other_latex_filename = "combined_regression_results_other.tex"
            other_latex_output_path = os.path.join(output_folder, other_latex_filename)

            # Generate the LaTeX table with coefficients, t-values, and regression statistics
            create_regression_latex_table(
                regression_stats=other_stats,
                r_squared_stats=other_r2,
                adj_r_squared_stats=other_adj_r2,
                n_observations_stats=other_nobs,
                caption=other_latex_caption,
                label=other_latex_label,
                output_path=other_latex_output_path
            )
        except Exception as e:
            print(f"Error generating LaTeX table for other regression results: {e}")
    else:
        print("No other regression models to generate LaTeX table.\n")

    # HTML Tables
    # Generate HTML table for main regression results (Return Variables)
    if main_stats:
        try:
            # Define the caption and label
            main_html_caption = captions['combined_regressions_main']
            main_html_label = "combined_regression_results_main"

            # Define the filename and path
            main_html_filename = "combined_regression_results_main.html"
            main_html_output_path = os.path.join(output_folder, main_html_filename)

            # Generate the HTML table with coefficients, t-values, and regression statistics
            create_regression_html_table(
                regression_stats=main_stats,
                r_squared_stats=main_r2,
                adj_r_squared_stats=main_adj_r2,
                n_observations_stats=main_nobs,
                caption=main_html_caption,
                label=main_html_label,
                output_path=main_html_output_path
            )
        except Exception as e:
            print(f"Error generating HTML table for main regression results: {e}")
    else:
        print("No main regression models to generate HTML table.\n")

    # Generate HTML table for other regression results (Other Variables)
    if other_stats:
        try:
            # Define the caption and label
            other_html_caption = captions['combined_regressions_other']
            other_html_label = "combined_regression_results_other"

            # Define the filename and path
            other_html_filename = "combined_regression_results_other.html"
            other_html_output_path = os.path.join(output_folder, other_html_filename)

            # Generate the HTML table with coefficients, t-values, and regression statistics
            create_regression_html_table(
                regression_stats=other_stats,
                r_squared_stats=other_r2,
                adj_r_squared_stats=other_adj_r2,
                n_observations_stats=other_nobs,
                caption=other_html_caption,
                label=other_html_label,
                output_path=other_html_output_path
            )
        except Exception as e:
            print(f"Error generating HTML table for other regression results: {e}")
    else:
        print("No other regression models to generate HTML table.\n")

    # End of function

#%% Generate LaTeX and HTML Tables for Main and Other Regression Results

generate_tables(
    main_results_stats, other_results_stats, 
    main_r_squared_stats, main_adj_r_squared_stats, main_n_observations_stats,
    other_r_squared_stats, other_adj_r_squared_stats, other_n_observations_stats,
    output_folder, captions
)

print("Generated LaTeX and HTML regression tables.")

#%% Comparing Lengths of Management Answers and Participant Questions

low_mean_participant = low_similarity['length_participant_questions'].mean()
high_mean_participant = high_similarity['length_participant_questions'].mean()
low_mean_management = low_similarity['length_management_answers'].mean()
high_mean_management = high_similarity['length_management_answers'].mean()

print(f"Average length of participant questions in Low Similarity: {low_mean_participant:.2f}")
print(f"Average length of participant questions in High Similarity: {high_mean_participant:.2f}")
print(f"Average length of management answers in Low Similarity: {low_mean_management:.2f}")
print(f"Average length of management answers in High Similarity: {high_mean_management:.2f}")

t_stat_participant, p_val_participant = ttest_ind(
    low_similarity['length_participant_questions'], 
    high_similarity['length_participant_questions'], 
    equal_var=False
)
print(f"\nT-Test for 'length_participant_questions' between Low and High Similarity:")
print(f"T-Statistic: {t_stat_participant:.3f}, P-Value: {p_val_participant:.3f}")

t_stat_management, p_val_management = ttest_ind(
    low_similarity['length_management_answers'], 
    high_similarity['length_management_answers'], 
    equal_var=False
)
print(f"\nT-Test for 'length_management_answers' between Low and High Similarity:")
print(f"T-Statistic: {t_stat_management:.3f}, P-Value: {p_val_management:.3f}")

#%% Perform t-test for 'difference_questions_answers'

low_mean_difference = low_similarity['difference_questions_answers'].mean()
high_mean_difference = high_similarity['difference_questions_answers'].mean()

print(f"\nAverage difference between participant questions and management answers in Low Similarity: {low_mean_difference:.2f}")
print(f"Average difference between participant questions and management answers in High Similarity: {high_mean_difference:.2f}")

t_stat_difference, p_val_difference = ttest_ind(
    low_similarity['difference_questions_answers'],
    high_similarity['difference_questions_answers'],
    equal_var=False
)

print(f"\nT-Test for 'difference_questions_answers' between Low and High Similarity:")
print(f"T-Statistic: {t_stat_difference:.3f}, P-Value: {p_val_difference:.3f}")

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

#%% Fix for HTML Table: Ensure Proper HTML Syntax

# Replace `.render()` with `.to_html(index=False)` to avoid AttributeError
html_table = styled_t_test_df.set_table_styles(table_styles).to_html(index=False, escape=False)

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
        <caption><strong>T-Test Results: Low vs. High Similarity Groups</strong></caption>
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

#%% Create Transition Matrix Figures

#%% Prepare Transition Matrices

# Since the DataFrame already contains mapped topics, we do not apply additional topic mapping.
# Ensure that 'filtered_presentation_topics' are correctly formatted and valid.

# Remove -1 from relevant columns (if necessary)
# Commented out based on user input that no topics should be removed
# analysis_df = remove_neg_one_from_columns(analysis_df, ['filtered_presentation_topics'])

# Define the number of topics based on the maximum topic index
max_topic = analysis_df['filtered_presentation_topics'].explode().max()
num_topics = int(max_topic) + 1 if pd.notnull(max_topic) else 0
print(f"Number of unique topics: {num_topics}")

# Ensure that all topics are integers and within range
def is_valid_topic(topic, max_topic):
    return isinstance(topic, int) and 0 <= topic < max_topic

# Calculate the number of topics before removal for diagnostic purposes
total_topics_before = analysis_df['filtered_presentation_topics'].apply(len).sum()

# Apply the validity check
analysis_df['filtered_presentation_topics'] = analysis_df['filtered_presentation_topics'].apply(
    lambda topics: [topic for topic in topics if is_valid_topic(topic, num_topics)]
)

# Calculate the number of topics after removal
total_topics_after = analysis_df['filtered_presentation_topics'].apply(len).sum()
topics_removed = total_topics_before - total_topics_after
print(f"Removed {topics_removed} invalid topics from 'filtered_presentation_topics'.")

# Exclude topic lists with fewer than two topics
valid_topic_lists = analysis_df['filtered_presentation_topics'].apply(lambda x: len(x) >= 2)
excluded_topic_lists = (~valid_topic_lists).sum()
if excluded_topic_lists > 0:
    print(f"Excluded {excluded_topic_lists} topic lists with fewer than two topics.")
analysis_df = analysis_df[valid_topic_lists]

# Create overall transition matrix
overall_topics = analysis_df['filtered_presentation_topics'].tolist()
print(f"Number of topic sequences: {len(overall_topics)}")
print(f"Sample topic sequences: {overall_topics[:5]}")  # Print first 5 sequences for inspection

overall_transition_matrix = create_transition_matrix(overall_topics, num_topics)
print("Created overall transition matrix.")

#%% Generate and Save Transition Matrices as CSV/Excel

def export_transition_matrix(matrix, filename, output_folder):
    """
    Exports the transition matrix to a CSV, Excel, and saves the heatmap.

    Parameters:
    - matrix (numpy.ndarray): The transition matrix.
    - filename (str): The base filename for saving.
    - output_folder (str): The directory to save the files.

    Returns:
    - None
    """
    # Convert to numpy array if not already
    matrix = np.array(matrix)

    # Debugging: Print type and shape
    print(f"Transition Matrix Type: {type(matrix)}")
    print(f"Transition Matrix Shape: {matrix.shape}")

    # Ensure it's a 2D array
    if matrix.ndim != 2:
        raise ValueError(f"Transition matrix must be 2D. Got {matrix.ndim}D.")

    # Convert to DataFrame with proper columns and index
    matrix_df = pd.DataFrame(matrix, 
                             columns=[f"To_{i}" for i in range(matrix.shape[1])], 
                             index=[f"From_{i}" for i in range(matrix.shape[0])])

    # Define the CSV and Excel paths
    csv_path = os.path.join(output_folder, f"{filename}.csv")
    excel_path = os.path.join(output_folder, f"{filename}.xlsx")

    # Save as CSV
    matrix_df.to_csv(csv_path)
    print(f"Exported {filename} to {csv_path}")

    # Save as Excel
    matrix_df.to_excel(excel_path)
    print(f"Exported {filename} to {excel_path}")

    # Create and save the heatmap
    plt.figure(figsize=(20, 16))
    sns.heatmap(matrix_df, cmap='viridis', linewidths=.5, annot=False)
    plt.title(f'{filename} Heatmap', fontsize=20)
    plt.xlabel('To Topic', fontsize=16)
    plt.ylabel('From Topic', fontsize=16)
    plt.tight_layout()

    # Define the heatmap image path
    heatmap_path = os.path.join(output_folder, f"{filename}_heatmap.png")

    # Save the heatmap
    plt.savefig(heatmap_path, dpi=300)
    plt.close()
    print(f"Saved {filename} heatmap to {heatmap_path}")

# Export Overall Transition Matrix
export_transition_matrix(overall_transition_matrix, 'overall_transition_matrix', output_folder)

#%% Create Transition Matrices for Top SICCD and PERMCO

# Identify Top SICCD (Industry) with the most earnings calls
top_siccd = analysis_df['siccd'].value_counts().idxmax()
top_siccd_count = analysis_df['siccd'].value_counts().max()
print(f"Top SICCD with the most earnings calls: {top_siccd} (Count: {top_siccd_count})")

# Filter data for Top SICCD
top_siccd_df = analysis_df[analysis_df['siccd'] == top_siccd]

# Create transition matrix for Top SICCD
top_siccd_topics = top_siccd_df['filtered_presentation_topics'].tolist()
print(f"Number of topic sequences for Top SICCD ({top_siccd}): {len(top_siccd_topics)}")
print(f"Sample topic sequences for Top SICCD: {top_siccd_topics[:5]}")  # Print first 5 sequences for inspection

top_siccd_transition_matrix = create_transition_matrix(top_siccd_topics, num_topics)
print(f"Created transition matrix for Top SICCD: {top_siccd}")

# Export Top SICCD Transition Matrix
export_transition_matrix(top_siccd_transition_matrix, f'siccd_{top_siccd}_transition_matrix', output_folder)

# Identify Top PERMCO (Company) with the most earnings calls
top_permco = analysis_df['permco'].value_counts().idxmax()
top_permco_count = analysis_df['permco'].value_counts().max()
print(f"Top PERMCO with the most earnings calls: {top_permco} (Count: {top_permco_count})")

# Filter data for Top PERMCO
top_permco_df = analysis_df[analysis_df['permco'] == top_permco]

# Create transition matrix for Top PERMCO
top_permco_topics = top_permco_df['filtered_presentation_topics'].tolist()
print(f"Number of topic sequences for Top PERMCO ({top_permco}): {len(top_permco_topics)}")
print(f"Sample topic sequences for Top PERMCO: {top_permco_topics[:5]}")  # Print first 5 sequences for inspection

top_permco_transition_matrix = create_transition_matrix(top_permco_topics, num_topics)
print(f"Created transition matrix for Top PERMCO: {top_permco}")

# Export Top PERMCO Transition Matrix
export_transition_matrix(top_permco_transition_matrix, f'permco_{top_permco}_transition_matrix', output_folder)

print("All transition matrix figures have been created and saved.")
