# read csv file from filepath
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import statsmodels.api as sm
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_squared_error
from scipy.stats import chi2_contingency, ttest_ind

# Path to the final dataset
filepath = "D:/daten_masterarbeit/final_dataset_reg_full.csv"

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
    'epsfxq_next',
    'length_participant_questions',  # Neue abhängige Variable
    'length_management_answers',    # Neue abhängige Variable
    'market_cap',                   # Kontrollvariable
    'rolling_beta',                  # Kontrollvariable
    'ceo_participates',             # Kontrollvariable
    'ceo_cfo_change',               # Kontrollvariable
    'word_length_presentation',     # Kontrollvariable
    'participant_question_topics',  # Für Chi-Quadrat-Test
    'management_answer_topics'      # Für Chi-Quadrat-Test
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

# Neue abhängige Variablen für zusätzliche Analysen
additional_dependent_vars = [
    'length_participant_questions',
    'length_management_answers'
]

# Kontrollvariablen
control_vars = [
    'market_cap',
    'ceo_participates',
    'ceo_cfo_change',
    'word_length_presentation'
]

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

# Funktion zur Durchführung der Regression mit statsmodels und Anzeige der p-Werte
def perform_regression_with_statsmodels(y_var, x_vars):
    X = analysis_df[x_vars]
    y = analysis_df[y_var]
    X = sm.add_constant(X)  # Fügt einen konstanten Term zum Prädiktor hinzu
    model = sm.OLS(y, X).fit()
    print(f"Regression results for {y_var} ~ {', '.join(x_vars)}:")
    print(model.summary())  # Enthält p-Werte, R-squared und andere Statistiken
    print("\n")
    return model

# Perform regression for each return variable on similarity variables
for ret_var in return_vars:
    perform_regression_with_statsmodels(ret_var, similarity_vars)

#%% Linear Regression using Scikit-learn

# Funktion zur Durchführung der Regression und Anzeige der Ergebnisse
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

#%% Zusätzliche Regressionen mit Kontrollvariablen

# Liste der zusätzlichen abhängigen Variablen
additional_dependent_vars = [
    'length_participant_questions',
    'length_management_answers'
]

for dependent_var in additional_dependent_vars:
    independent_vars = similarity_vars + control_vars
    print(f"Regression results for {dependent_var} ~ {', '.join(independent_vars)}:")
    perform_regression_with_statsmodels(dependent_var, independent_vars)
    perform_sklearn_regression(dependent_var, independent_vars)

#%% Chi-Quadrat-Tests für Themenverteilungen

# Funktion zur Durchführung des Chi-Quadrat-Tests
def perform_chi_squared_test(group, category_var1, category_var2):
    contingency_table = pd.crosstab(group[category_var1], group[category_var2])
    chi2, p, dof, ex = chi2_contingency(contingency_table)
    print(f"Chi-Quadrat-Test zwischen '{category_var1}' und '{category_var2}':")
    print(f"Chi2: {chi2}, p-Wert: {p}, Freiheitsgrade: {dof}")
    print("Kontingenztabelle:")
    print(contingency_table)
    print("\n")
    return chi2, p

# Gruppierung basierend auf 'similarity_to_overall_average'
# Definieren von Quantilen, z.B. oberste 20% vs unterste 20%
analysis_df['similarity_group'] = pd.qcut(analysis_df['similarity_to_overall_average'], 
                                         q=5, 
                                         labels=['Very Low', 'Low', 'Medium', 'High', 'Very High'])

# Separate Gruppen
low_similarity = analysis_df[analysis_df['similarity_group'].isin(['Very Low', 'Low'])]
high_similarity = analysis_df[analysis_df['similarity_group'].isin(['High', 'Very High'])]

# Durchführung des Chi-Quadrat-Tests für beide Gruppen
print("Chi-Quadrat-Test für niedrige Similarity Gruppe:")
perform_chi_squared_test(low_similarity, 'participant_question_topics', 'management_answer_topics')

print("Chi-Quadrat-Test für hohe Similarity Gruppe:")
perform_chi_squared_test(high_similarity, 'participant_question_topics', 'management_answer_topics')

#%% Vergleich der Anzahl der Managementantworten

# Annahme: Es gibt eine Spalte 'length_management_answers' im DataFrame
# Falls nicht vorhanden, hier ein Beispiel, wie es erstellt werden könnte
# Beispiel: Anzahl der Managementantworten pro Call
# analysis_df['length_management_answers'] = df.groupby('call_id')['management_answer'].transform('count')

# Gruppierung der Daten
# low_similarity_group = analysis_df[analysis_df['similarity_group'].isin(['Very Low', 'Low'])]
# high_similarity_group = analysis_df[analysis_df['similarity_group'].isin(['High', 'Very High'])]

# Vergleich der durchschnittlichen Anzahl der Managementantworten
low_mean_participant = low_similarity['length_participant_questions'].mean()
high_mean_participant = high_similarity['length_participant_questions'].mean()
low_mean_management = low_similarity['length_management_answers'].mean()
high_mean_management = high_similarity['length_management_answers'].mean()

print(f"Durchschnittliche Länge der Teilnehmerfragen bei niedriger Similarity: {low_mean_participant}")
print(f"Durchschnittliche Länge der Teilnehmerfragen bei hoher Similarity: {high_mean_participant}")
print(f"Durchschnittliche Länge der Managementantworten bei niedriger Similarity: {low_mean_management}")
print(f"Durchschnittliche Länge der Managementantworten bei hoher Similarity: {high_mean_management}")

# Statistischer Test (z.B. t-Test) für 'length_participant_questions'
t_stat_participant, p_val_participant = ttest_ind(
    low_similarity['length_participant_questions'], 
    high_similarity['length_participant_questions'], 
    equal_var=False
)
print(f"T-Test für die Länge der Teilnehmerfragen zwischen niedriger und hoher Similarity:")
print(f"T-Statistik: {t_stat_participant}, p-Wert: {p_val_participant}\n")

# Statistischer Test (z.B. t-Test) für 'length_management_answers'
t_stat_management, p_val_management = ttest_ind(
    low_similarity['length_management_answers'], 
    high_similarity['length_management_answers'], 
    equal_var=False
)
print(f"T-Test für die Länge der Managementantworten zwischen niedriger und hoher Similarity:")
print(f"T-Statistik: {t_stat_management}, p-Wert: {p_val_management}\n")

#%% Zusätzliche Visualisierungen

# Beispiel: Gestapelte Balkendiagramme für Themenverteilungen
def plot_topic_distribution(group, title):
    topic_counts = pd.crosstab(group['participant_question_topics'], group['management_answer_topics'])
    topic_counts.plot(kind='bar', stacked=True, figsize=(10,7))
    plt.title(title)
    plt.xlabel('Participant Question Topics')
    plt.ylabel('Anzahl der Managementantworten')
    plt.legend(title='Management Answer Topics')
    plt.show()

plot_topic_distribution(low_similarity, 'Themenverteilung - Niedrige Similarity')
plot_topic_distribution(high_similarity, 'Themenverteilung - Hohe Similarity')

#%% Speichern der Ergebnisse (optional)

# Beispiel: Speichern der Regressionsergebnisse oder Chi-Quadrat-Testergebnisse in Dateien
# with open('regression_results.txt', 'w') as f:
#     f.write(model.summary().as_text())

#%% Hinweise zur Interpretation

"""
- Die Regressionsergebnisse zeigen den Einfluss der similarity Variablen auf die abhängigen Variablen, wobei Kontrollvariablen berücksichtigt werden.
- Die Chi-Quadrat-Tests überprüfen, ob es einen signifikanten Zusammenhang zwischen den Themen der Analystenfragen und den Managementantworten gibt, getrennt nach Similarity-Gruppen.
- Der Vergleich der Länge der Managementantworten und Teilnehmerfragen zwischen den Gruppen hilft zu verstehen, ob die Länge der Antworten von der Similarity beeinflusst wird.
"""
