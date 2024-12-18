# variable_analysis.py

# Import necessary libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import statsmodels.api as sm
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
# Ensure all variables exist in the DataFrame   
missing_vars = [var for var in variables if var not in df.columns]
if missing_vars:
    raise KeyError(f"The following required columns are missing from the DataFrame: {missing_vars}")

analysis_df = df[variables].dropna()
print(f"Number of observations after dropping NaNs: {len(analysis_df)}")

# Parse Topic Columns
def parse_topics(topics):
    """Parses a topic column into a list of integers.

    Args:
        topics: A string or list of topics, where each topic is an integer.

    Returns:
        A list of integers representing the topics.

    Raises:
        ValueError: If the input string cannot be parsed as a list of integers.
        SyntaxError: If the input string has invalid syntax.
    """
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
    """
    Flattens a list of topics into a single list.

    Args:
        topics: A list of topics, where each topic may be a list itself.

    Returns:
        A single list containing all topics. If a topic is a list, its elements
        are added individually to the returned list.

    Example:
        flatten_topics([[1, 2], 3, [4, 5]]) returns [1, 2, 3, 4, 5]
    """

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
    """
    Converts a list of topics into a list of integers.

    Args:
        topics: A list of topics, where each topic may be an integer, string, or list.

    Returns:
        A list of integers representing the topics. If a topic is a list, its elements
        are added individually to the returned list. If a topic is a string and cannot
        be parsed as an integer, it is ignored.

    Example:
        convert_topics_to_int([[1, 2], '3', [4, '5']]) returns [1, 2, 3, 4, 5]
    """

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
# Apply the functions
topic_columns = ['filtered_presentation_topics', 'participant_question_topics', 'management_answer_topics']
for col in topic_columns:
    analysis_df[col] = analysis_df[col].apply(parse_topics)
    analysis_df[col] = analysis_df[col].apply(convert_topics_to_int)

def verify_topics_are_int(topics):
    """
    Verifies that all elements in a list are integers.

    Args:
        topics (list): The list of topics to verify.

    Returns:
        bool: True if all elements in the list are integers, False otherwise.
    """
    return all(isinstance(topic, int) for topic in topics)
# Verify that all topics are integers
verification = analysis_df[topic_columns].applymap(verify_topics_are_int).all(axis=1)
if not verification.all():
    problematic_rows = verification[verification == False].index.tolist()
    print(f"Warning: The following rows have non-integer topics and will be excluded: {problematic_rows}")
    analysis_df = analysis_df[verification]
# Check if all topic columns are correctly formatted as lists of integers
else:
    print("All topic columns are correctly formatted as lists of integers.")

print("\nSample 'filtered_presentation_topics' after parsing:")
print(analysis_df['filtered_presentation_topics'].head())

print("\nNumber of topics per row:")
print(analysis_df['filtered_presentation_topics'].apply(len).describe())

# Count unique topics
unique_topics = analysis_df['filtered_presentation_topics'].explode().unique()
print(f"\nNumber of unique topics after parsing: {len(unique_topics)}")
print(f"Unique topics: {unique_topics}")

# Create 'topic_diversity' Variable
def calculate_topic_diversity(topics):
    """
    Calculates the topic diversity of a list of topics.

    Topic diversity is defined as the number of unique topics divided by the total number of topics.

    Args:
        topics (list): A list of topics, where each topic is an integer.

    Returns:
        float: The topic diversity of the list of topics. If the input is not a list or is empty, returns NaN.

    Example:
        calculate_topic_diversity([1, 2, 3]) returns 1.0
    """


    if not isinstance(topics, list) or len(topics) == 0:
        return np.nan
    unique_topics = set(topics)
    diversity = len(unique_topics) / len(topics)
    return diversity

# Apply the function to create the 'topic_diversity' variable
analysis_df['topic_diversity'] = analysis_df['filtered_presentation_topics'].apply(calculate_topic_diversity)
mean_diversity = analysis_df['topic_diversity'].mean()
analysis_df['topic_diversity'] = analysis_df['topic_diversity'].fillna(mean_diversity)
print("Created 'topic_diversity' variable.")

# Descriptive Statistics
descriptive_stats_folder = os.path.join(output_folder, "descriptive_statistics")
if not os.path.exists(descriptive_stats_folder):
    os.makedirs(descriptive_stats_folder)
    print(f"Created folder for descriptive statistics: {descriptive_stats_folder}")
else:
    print(f"Folder for descriptive statistics already exists: {descriptive_stats_folder}")

# Numerical Descriptive Statistics
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

# Save LaTeX table
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

control_vars = [
    'market_cap',
    'rolling_beta',
    'word_length_presentation',
    'ceo_participates',
    'ceo_cfo_change'
]

# Standardize independent and control variables
variables_to_scale = independent_vars + ['market_cap', 'rolling_beta', 'word_length_presentation']

scaler = StandardScaler()
analysis_df[variables_to_scale] = scaler.fit_transform(analysis_df[variables_to_scale])
print(f"Standardized variables: {', '.join(variables_to_scale)}")

def calculate_vif(vars_list, df):
    """
    Calculates the Variance Inflation Factor (VIF) for a given list of variables.

    VIF is a measure used to detect multicollinearity in a set of multiple regression variables. 
    A high VIF indicates that the associated variable is highly collinear with other variables.

    Args:
        vars_list (list): A list of strings representing the variable names for which to calculate VIF.
        df (pandas.DataFrame): The DataFrame containing the variables.

    Returns:
        pandas.DataFrame: A DataFrame containing the variable names and their corresponding VIF values.
    """

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

# Define Regression Groups
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
        # originally contained 'difference_questions_answers', removed as not needed
    ]
}

print("Defined regression groups.")

# Define Captions for Regression Tables
captions = {
    'combined_regressions_main': "Combined OLS Regression Results for Return Variables",
    'combined_regressions_other': "Combined OLS Regression Results for Other Variables"
}

print("Defined captions for regression tables.")

def clean_variable(var):
    """
    Clean a variable name by removing newline characters, the 'Name: x, dtype: object' prefix, and any leading/trailing whitespace.
    
    This is necessary because pandas sometimes returns variable names with the above format, and it's not desirable to display this in LaTeX output.
    
    Parameters
    ----------
    var : str or object
        The variable to clean. If not a string, will be converted to one with str() and then cleaned.
    
    Returns
    -------
    str
        The cleaned variable name.
    """
    if isinstance(var, str):
        var = var.split('\n')[0]
        var = re.sub(r'Name:\s*\d+,\s*dtype:\s*object', '', var)
        var = var.strip()
        return var
    else:
        return str(var).strip()

def create_regression_latex_table(regression_stats, r_squared_stats, adj_r_squared_stats, n_observations_stats, caption, label, output_path):
    """
        Generate a LaTeX table for regression results.

        This function creates a LaTeX formatted table from regression statistics, including coefficients, 
        t-values, R-squared, adjusted R-squared, and the number of observations. The table is saved to the 
        specified output path.

        Args:
            regression_stats (dict): A dictionary where the keys are model names and the values are DataFrames
                containing regression statistics such as coefficients and t-values.
            r_squared_stats (dict): A dictionary containing R-squared values for each model.
            adj_r_squared_stats (dict): A dictionary containing adjusted R-squared values for each model.
            n_observations_stats (dict): A dictionary containing the number of observations for each model.
            caption (str): The caption for the LaTeX table.
            label (str): The label for the LaTeX table, used for referencing.
            output_path (str): The file path where the LaTeX table will be saved.

        Writes:
            A LaTeX file containing the regression table to the specified output path.
    """

    combined_df = pd.DataFrame()

    # Concatenate DataFrames for each model
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

    column_alignment = 'l' + 'c' * len(regression_stats.keys())

    # Generate LaTeX content
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

    for i, row in formatted_df.iterrows():
        variable = row['Variable']
        if variable == '':
            latex_content += "                            & " + " & ".join(row[1:]) + " \\\\\n"
        else:
            variable_escaped = re.sub(r'([&_#%{}~^\\])', r'\\\1', variable)
            latex_content += f"    {variable_escaped} & " + " & ".join(row[1:]) + " \\\\\n"

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

    latex_content += f"""
    \\hline
    \\end{{tabular}}}}
    }}
    \\bigskip
    \\textit{{t-values in parentheses.}} \\ 
    * p<.1, ** p<.05, ***p<.01
\\end{{table}}
"""

    with open(output_path, 'w', encoding='utf-8') as f:
        f.write(latex_content)

    print(f"Saved LaTeX regression table to {output_path}")

def create_regression_html_table(regression_stats, r_squared_stats, adj_r_squared_stats, n_observations_stats, caption, label, output_path):
    """
    Generate an HTML table for regression results.

    This function creates an HTML table from regression statistics, including coefficients, 
    t-values, R-squared, adjusted R-squared, and the number of observations. The table is saved to the 
    specified output path.

    Parameters
    ----------
    regression_stats : dict
        A dictionary where the keys are model names and the values are DataFrames
        containing regression statistics such as coefficients and t-values.
    r_squared_stats : dict
        A dictionary containing R-squared values for each model.
    adj_r_squared_stats : dict
        A dictionary containing adjusted R-squared values for each model.
    n_observations_stats : dict
        A dictionary containing the number of observations for each model.
    caption : str
        The caption for the HTML table.
    label : str
        The label for the HTML table, used for referencing.
    output_path : str
        The file path where the HTML table will be saved.

    Writes
    ------
    An HTML file containing the regression table to the specified output path.
    """
    combined_df = pd.DataFrame()

    # Concatenate DataFrames for each model
    for model_name, stats in regression_stats.items():
        stats = stats.copy()
        stats.columns = pd.MultiIndex.from_product([[model_name], stats.columns])
        combined_df = pd.concat([combined_df, stats], axis=1)
    
    # Reset index
    combined_df.reset_index(inplace=True)
    combined_df.rename(columns={'index': 'Variable'}, inplace=True)
    combined_df['Variable'] = combined_df['Variable'].apply(clean_variable)

    # Generate HTML content
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

    headers = ['Variable'] + list(regression_stats.keys())
    formatted_df = pd.DataFrame(rows, columns=headers)
    formatted_df['Variable'] = formatted_df['Variable'].astype(str)

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

    for i, row in formatted_df.iterrows():
        variable = row['Variable']
        if variable == '':
            html_content += "<tr>\n"
            html_content += "    <td></td>" + "".join([f"<td>{cell}</td>" for cell in row[1:]]) + "\n</tr>\n"
        else:
            variable_escaped = variable.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
            html_content += f"<tr>\n    <td>{variable_escaped}</td>" + "".join([f"<td>{cell}</td>" for cell in row[1:]]) + "\n</tr>\n"

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

    with open(output_path, 'w', encoding='utf-8') as f:
        f.write(html_content)

    print(f"Saved HTML regression table to {output_path}")

def perform_combined_regressions(regression_groups, analysis_df, independent_vars, control_vars, captions):
    """
    Perform a set of regressions with multiple dependent variables and multiple groups, and return the results.

    Parameters
    ----------
    regression_groups : dict
        A dictionary where the keys are the group names and the values are lists of dependent variable names.
    analysis_df : pandas.DataFrame
        The DataFrame containing the data to be analyzed.
    independent_vars : list
        A list of independent variable names to include in the regression.
    control_vars : list
        A list of control variable names to include in the regression.
    captions : list
        A list of captions to be used for the HTML table.

    Returns
    -------
    main_results_stats : dict
        A dictionary where the keys are the dependent variable names and the values are the regression results.
    other_results_stats : dict
        A dictionary where the keys are the dependent variable names and the values are the regression results.
    main_r_squared_stats : dict
        A dictionary where the keys are the dependent variable names and the values are the R-squared values.
    main_adj_r_squared_stats : dict
        A dictionary where the keys are the dependent variable names and the values are the adjusted R-squared values.
    main_n_observations_stats : dict
        A dictionary where the keys are the dependent variable names and the values are the number of observations.
    other_r_squared_stats : dict
        A dictionary where the keys are the dependent variable names and the values are the R-squared values.
    other_adj_r_squared_stats : dict
        A dictionary where the keys are the dependent variable names and the values are the adjusted R-squared values.
    other_n_observations_stats : dict
        A dictionary where the keys are the dependent variable names and the values are the number of observations.
    """
    # Initialize dictionaries to hold the results
    main_results_stats = {}
    other_results_stats = {}
    
    main_r_squared_stats = {}
    main_adj_r_squared_stats = {}
    main_n_observations_stats = {}
    
    other_r_squared_stats = {}
    other_adj_r_squared_stats = {}
    other_n_observations_stats = {}
    
    rename_mapping_main_latex = {
        'excess_ret_immediate': 'r_immediate',
        'excess_ret_short_term': 'r_short',
        'excess_ret_medium_term': 'r_medium',
        'excess_ret_long_term': 'r_long',
        'epsfxq': 'eps',
        'epsfxq_next': 'eps_next'
    }
    
    # List of models to return
    return_models = ['r_immediate', 'r_short', 'r_medium', 'r_long']
    
    rename_mapping_other = {
        'length_participant_questions': 'length_participant_questions',
        'length_management_answers': 'length_management_answers'
    }
    
    # Perform regressions
    for group_name, dep_vars in regression_groups.items():
        for dep_var in dep_vars:
            independent_vars_combined = independent_vars + control_vars
            X = analysis_df[independent_vars_combined]
            y = analysis_df[dep_var]
            X = sm.add_constant(X)

            try:
                model = sm.OLS(y, X).fit()
                summary_df = model.summary2().tables[1]

                if {'Coef.', 't', 'P>|t|'}.issubset(summary_df.columns):
                    summary_df = summary_df[['Coef.', 't', 'P>|t|']]
                elif {'Coef.', 't-value', 'P>|t|'}.issubset(summary_df.columns):
                    summary_df = summary_df[['Coef.', 't-value', 'P>|t|']]
                else:
                    raise KeyError(f"Expected columns for {dep_var} not found in summary table.")

                summary_df = summary_df.rename(columns={
                    'Coef.': 'Coefficient',
                    't': 'stat_value',
                    't-value': 'stat_value',
                    'P>|t|': 'p_value'
                })
                # Add significance stars
                def add_significance_stars(p):
                    if p < 0.001:
                        return '***'
                    elif p < 0.01:
                        return '**'
                    elif p < 0.05:
                        return '*'
                    else:
                        return ''
                # Rename dependent variable
                if dep_var in rename_mapping_main_latex:
                    short_name_latex = rename_mapping_main_latex[dep_var]
                    format_decimal = 4 if short_name_latex in return_models else 3
                else:
                    short_name_latex = rename_mapping_other.get(dep_var, dep_var)
                    format_decimal = 3
                # Format the values
                if short_name_latex in return_models:
                    summary_df['Coefficient'] = summary_df.apply(
                        lambda row: f"{row['Coefficient']:.4f}{add_significance_stars(row['p_value'])}", axis=1
                    )
                else:
                    summary_df['Coefficient'] = summary_df.apply(
                        lambda row: f"{row['Coefficient']:.3f}{add_significance_stars(row['p_value'])}", axis=1
                    )
                # Format the t-value
                summary_df['stat_value'] = summary_df['stat_value'].apply(lambda x: f"({x:.3f})")
                summary_df = summary_df[['Coefficient', 'stat_value']]
                
                # Save the results
                if group_name == 'return_vars' and dep_var in rename_mapping_main_latex:
                    main_results_stats[short_name_latex] = summary_df
                    main_r_squared_stats[short_name_latex] = model.rsquared
                    main_adj_r_squared_stats[short_name_latex] = model.rsquared_adj
                    main_n_observations_stats[short_name_latex] = int(model.nobs)
                else:
                    short_name = rename_mapping_other.get(dep_var, dep_var)
                    other_results_stats[short_name] = summary_df
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
    # Return the results
    return (main_results_stats, other_results_stats,
            main_r_squared_stats, main_adj_r_squared_stats, main_n_observations_stats,
            other_r_squared_stats, other_adj_r_squared_stats, other_n_observations_stats)

# Perform combined regressions
print("Performing combined regressions...")
(main_results_stats, other_results_stats,
 main_r_squared_stats, main_adj_r_squared_stats, main_n_observations_stats,
 other_r_squared_stats, other_adj_r_squared_stats, other_n_observations_stats) = perform_combined_regressions(
    regression_groups, analysis_df, independent_vars, control_vars, captions
)
print("Performed combined regressions and captured results.")

def generate_tables(main_stats, other_stats, 
                    main_r2, main_adj_r2, main_nobs, 
                    other_r2, other_adj_r2, other_nobs,
                    output_folder, captions):
    # LaTeX Tables for Main
    """
    Generate LaTeX and HTML tables for regression results.

    This function creates LaTeX and HTML tables for both main and other regression
    models using the provided statistics. The tables are saved to the specified
    output folder.

    Parameters:
    - main_stats: Dictionary of main regression statistics.
    - other_stats: Dictionary of other regression statistics.
    - main_r2: Dictionary of R-squared values for main regressions.
    - main_adj_r2: Dictionary of adjusted R-squared values for main regressions.
    - main_nobs: Dictionary of the number of observations for main regressions.
    - other_r2: Dictionary of R-squared values for other regressions.
    - other_adj_r2: Dictionary of adjusted R-squared values for other regressions.
    - other_nobs: Dictionary of the number of observations for other regressions.
    - output_folder: Path to the folder where the tables will be saved.
    - captions: Dictionary containing captions for the tables.

    The function attempts to create both LaTeX and HTML tables for the main and
    other regression models. If an error occurs during table generation, an error
    message is printed.
    """
    # LaTeX Tables for Main
    if main_stats:
        try:
            main_latex_caption = captions['combined_regressions_main']
            main_latex_label = "tab:combined_regression_results_main"
            main_latex_filename = "combined_regression_results_main.tex"
            main_latex_output_path = os.path.join(output_folder, main_latex_filename)
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

    # LaTeX Tables for Other
    if other_stats:
        try:
            other_latex_caption = captions['combined_regressions_other']
            other_latex_label = "tab:combined_regression_results_other"
            other_latex_filename = "combined_regression_results_other.tex"
            other_latex_output_path = os.path.join(output_folder, other_latex_filename)
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

    # HTML Tables for Main
    if main_stats:
        try:
            main_html_caption = captions['combined_regressions_main']
            main_html_label = "combined_regression_results_main"
            main_html_filename = "combined_regression_results_main.html"
            main_html_output_path = os.path.join(output_folder, main_html_filename)
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

    # HTML Tables for Other
    if other_stats:
        try:
            other_html_caption = captions['combined_regressions_other']
            other_html_label = "combined_regression_results_other"
            other_html_filename = "combined_regression_results_other.html"
            other_html_output_path = os.path.join(output_folder, other_html_filename)
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

# Generate LaTeX and HTML tables
generate_tables(
    main_results_stats, other_results_stats, 
    main_r_squared_stats, main_adj_r_squared_stats, main_n_observations_stats,
    other_r_squared_stats, other_adj_r_squared_stats, other_n_observations_stats,
    output_folder, captions
)

print("Generated LaTeX and HTML regression tables.")

#%% Transition Matrix Stuff

# Prepare Transition Matrices
max_topic = analysis_df['filtered_presentation_topics'].explode().max()
num_topics = int(max_topic) + 1 if pd.notnull(max_topic) else 0
print(f"Number of unique topics: {num_topics}")

def is_valid_topic(topic, max_topic):
    """
    Check if a topic is valid, i.e., if it is an integer between 0 and max_topic-1.

    Parameters
    ----------
    topic : int or any
        The topic to check.
    max_topic : int
        The maximum number of topics.

    Returns
    -------
    bool
        True if the topic is valid, False otherwise.
    """
    return isinstance(topic, int) and 0 <= topic < max_topic

# Count the total number of topics
total_topics_before = analysis_df['filtered_presentation_topics'].apply(len).sum()

# Filter out invalid topics
analysis_df['filtered_presentation_topics'] = analysis_df['filtered_presentation_topics'].apply(
    lambda topics: [topic for topic in topics if is_valid_topic(topic, num_topics)]
)
# Count the remaining number of topics
total_topics_after = analysis_df['filtered_presentation_topics'].apply(len).sum()
topics_removed = total_topics_before - total_topics_after
print(f"Removed {topics_removed} invalid topics from 'filtered_presentation_topics'.")

# Filter out topic lists with fewer than two topics
valid_topic_lists = analysis_df['filtered_presentation_topics'].apply(lambda x: len(x) >= 2)
excluded_topic_lists = (~valid_topic_lists).sum()
if excluded_topic_lists > 0:
    print(f"Excluded {excluded_topic_lists} topic lists with fewer than two topics.")
analysis_df = analysis_df[valid_topic_lists]

# Count the remaining number of topics
overall_topics = analysis_df['filtered_presentation_topics'].tolist()
print(f"Number of topic sequences: {len(overall_topics)}")
print(f"Sample topic sequences: {overall_topics[:5]}")

# Create Transition Matrices
overall_transition_matrix = create_transition_matrix(overall_topics, num_topics)
print("Created overall transition matrix.")

def export_transition_matrix(matrix, filename, output_folder):
    
    """
    Export a transition matrix to a CSV file, an Excel file, and a heatmap PNG file.

    Parameters
    ----------
    matrix : numpy.ndarray
        The transition matrix to export.
    filename : str
        The filename to use for the exported files (without extension).
    output_folder : str
        The folder to which the files should be exported.

    Returns
    -------
    None
    """
    # Convert matrix to DataFrame
    matrix = np.array(matrix)
    print(f"Transition Matrix Type: {type(matrix)}")
    print(f"Transition Matrix Shape: {matrix.shape}")
    # Check if matrix is 2D
    if matrix.ndim != 2:
        raise ValueError(f"Transition matrix must be 2D. Got {matrix.ndim}D.")
    # Convert matrix to DataFrame
    matrix_df = pd.DataFrame(matrix, 
                             columns=[f"To_{i}" for i in range(matrix.shape[1])], 
                             index=[f"From_{i}" for i in range(matrix.shape[0])])

    csv_path = os.path.join(output_folder, f"{filename}.csv")
    excel_path = os.path.join(output_folder, f"{filename}.xlsx")

    # Export
    matrix_df.to_csv(csv_path)
    print(f"Exported {filename} to {csv_path}")

    matrix_df.to_excel(excel_path)
    print(f"Exported {filename} to {excel_path}")
    
    # Heatmap
    plt.figure(figsize=(20, 16))
    sns.heatmap(matrix_df, cmap='viridis', linewidths=.5, annot=False)
    plt.title(f'{filename} Heatmap', fontsize=20)
    plt.xlabel('To Topic', fontsize=16)
    plt.ylabel('From Topic', fontsize=16)
    plt.tight_layout()

    heatmap_path = os.path.join(output_folder, f"{filename}_heatmap.png")
    plt.savefig(heatmap_path, dpi=300)
    plt.close()
    print(f"Saved {filename} heatmap to {heatmap_path}")

export_transition_matrix(overall_transition_matrix, 'overall_transition_matrix', output_folder)

# Top SICCD
top_siccd = analysis_df['siccd'].value_counts().idxmax()
top_siccd_count = analysis_df['siccd'].value_counts().max()
print(f"Top SICCD with the most earnings calls: {top_siccd} (Count: {top_siccd_count})")
top_siccd_df = analysis_df[analysis_df['siccd'] == top_siccd]

top_siccd_topics = top_siccd_df['filtered_presentation_topics'].tolist()
print(f"Number of topic sequences for Top SICCD ({top_siccd}): {len(top_siccd_topics)}")
print(f"Sample topic sequences for Top SICCD: {top_siccd_topics[:5]}")

top_siccd_transition_matrix = create_transition_matrix(top_siccd_topics, num_topics)
print(f"Created transition matrix for Top SICCD: {top_siccd}")
export_transition_matrix(top_siccd_transition_matrix, f'siccd_{top_siccd}_transition_matrix', output_folder)

# Top PERMCO
top_permco = analysis_df['permco'].value_counts().idxmax()
top_permco_count = analysis_df['permco'].value_counts().max()
print(f"Top PERMCO with the most earnings calls: {top_permco} (Count: {top_permco_count})")
top_permco_df = analysis_df[analysis_df['permco'] == top_permco]

top_permco_topics = top_permco_df['filtered_presentation_topics'].tolist()
print(f"Number of topic sequences for Top PERMCO ({top_permco}): {len(top_permco_topics)}")
print(f"Sample topic sequences for Top PERMCO: {top_permco_topics[:5]}")

top_permco_transition_matrix = create_transition_matrix(top_permco_topics, num_topics)
print(f"Created transition matrix for Top PERMCO: {top_permco}")
export_transition_matrix(top_permco_transition_matrix, f'permco_{top_permco}_transition_matrix', output_folder)

print("All requested files have been created.")
