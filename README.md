# Boston
It is my lab work for the course
import piplite
await piplite.install(['numpy', 'pandas'])
await piplite.install(['seaborn'])

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import scipy.stats as stats
import statsmodels.api as sm
from statsmodels.formula.api import ols
import io
from js import fetch

# Load the dataset
URL = 'https://cf-courses-data.s3.us.cloud-object-storage.appdomain.cloud/IBMDeveloperSkillsNetwork-ST0151EN-SkillsNetwork/labs/boston_housing.csv'
resp = await fetch(URL)
boston_url = io.BytesIO((await resp.arrayBuffer()).to_py())
boston_df = pd.read_csv(boston_url)

# Display first few rows
boston_df.head()

# Descriptive statistics
summary_stats = boston_df.describe()
print("Summary Statistics:\n", summary_stats)

# Additional correlation heatmap
plt.figure(figsize=(10, 8))
sns.heatmap(boston_df.corr(), annot=True, cmap='coolwarm', fmt='.2f')
plt.title('Correlation Heatmap of Boston Housing Data')
plt.show()

# Visualization: Histogram of Median Value of Owner-Occupied Homes
plt.figure(figsize=(8, 6))
sns.histplot(boston_df['MEDV'], bins=30, kde=True, color='blue')
plt.title('Distribution of Median Value of Owner-Occupied Homes')
plt.xlabel('Median Value ($1000s)')
plt.ylabel('Frequency')
plt.show()

# Visualization: Histogram for Charles River Variable
plt.figure(figsize=(6, 4))
sns.countplot(x='CHAS', data=boston_df, palette='viridis', hue='CHAS', legend=False)
plt.title('Distribution of Homes Near Charles River')
plt.xlabel('Charles River (0 = No, 1 = Yes)')
plt.ylabel('Count')
plt.show()

# Visualization: MEDV vs AGE grouped into categories
boston_df['AGE_group'] = pd.cut(boston_df['AGE'], bins=[0, 35, 70, 100], labels=['<=35', '35-70', '>=70'])
plt.figure(figsize=(8, 6))
sns.boxplot(x='AGE_group', y='MEDV', data=boston_df, palette='coolwarm')
plt.title('Median Value of Homes by Age Group')
plt.xlabel('Age Group')
plt.ylabel('Median Value ($1000s)')
plt.show()

# Scatter Plot: NOX vs INDUS
plt.figure(figsize=(8, 6))
sns.scatterplot(x='NOX', y='INDUS', data=boston_df, color='red')
plt.title('Scatter Plot: NOX vs INDUS')
plt.xlabel('Nitrogen Oxides Concentration')
plt.ylabel('Non-Retail Business Acres')
plt.show()

# Histogram for PTRATIO
plt.figure(figsize=(8, 6))
sns.histplot(boston_df['PTRATIO'], bins=20, kde=True, color='green')
plt.title('Distribution of Student-Teacher Ratio')
plt.xlabel('Pupil-Teacher Ratio')
plt.ylabel('Frequency')
plt.show()

# Additional Visualization: MEDV vs RM (Rooms per dwelling)
plt.figure(figsize=(8, 6))
sns.scatterplot(x='RM', y='MEDV', data=boston_df, color='purple')
plt.title('Scatter Plot: MEDV vs RM (Rooms per Dwelling)')
plt.xlabel('Average Number of Rooms')
plt.ylabel('Median Value ($1000s)')
plt.show()

# Statistical Test 1: T-Test for MEDV by CHAS
print("\nT-Test: Checking if there is a significant difference in MEDV based on CHAS")
medv_chas_0 = boston_df[boston_df['CHAS'] == 0]['MEDV']
medv_chas_1 = boston_df[boston_df['CHAS'] == 1]['MEDV']
t_stat, p_value = stats.ttest_ind(medv_chas_0, medv_chas_1, equal_var=False)
print(f'T-Test: t-statistic={t_stat:.3f}, p-value={p_value:.3f}')

# Statistical Test 2: ANOVA for MEDV by AGE Group
print("\nANOVA: Testing if there is a significant difference in MEDV across AGE groups")
anova_model = ols('MEDV ~ C(AGE_group)', data=boston_df).fit()
anova_table = sm.stats.anova_lm(anova_model, typ=2)
print(anova_table)

# Statistical Test 3: Pearson Correlation between NOX and INDUS
print("\nPearson Correlation: Checking relationship between NOX and INDUS")
corr, p_corr = stats.pearsonr(boston_df['NOX'], boston_df['INDUS'])
print(f'Pearson Correlation: correlation={corr:.3f}, p-value={p_corr:.3f}')

# Statistical Test 4: Regression Analysis of DIS on MEDV
print("\nRegression Analysis: Checking impact of DIS on MEDV")
X = sm.add_constant(boston_df['DIS'])
y = boston_df['MEDV']
reg_model = sm.OLS(y, X).fit()
print(reg_model.summary())

# Checking Multicollinearity using VIF
from statsmodels.stats.outliers_influence import variance_inflation_factor
variables = boston_df[['DIS', 'RM', 'LSTAT']]
variables = sm.add_constant(variables)
vif_data = pd.DataFrame()
vif_data['Feature'] = variables.columns
vif_data['VIF'] = [variance_inflation_factor(variables.values, i) for i in range(variables.shape[1])]
print("\nVariance Inflation Factor (VIF) Analysis:")
print(vif_data)
