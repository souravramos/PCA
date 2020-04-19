from subprocess import check_output
import os
import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
import seaborn as sns

pd.set_option('display.max_columns', 20)
print(next(os.walk('input'))[2])

df = pd.read_csv('input\\HR_comma_sep.csv')

# to get column names
columns_names = df.columns
#print('Columns names : ')
#print(columns_names)
##print(df.shape)
##print(df.head())
#print(df.corr())

# visualising correlation using seaborn library
correlation = df.corr()
fig1 = plt.figure(1, figsize=(10, 10))
ax = sns.heatmap(correlation, vmax=1, square=True, annot=True, cmap='cubehelix')
bottom, top = ax.get_ylim()
ax.set_ylim(bottom + 0.5, top - 0.5)
plt.title('Correlation between different features')
fig1.savefig('Correlation between different features.png')
plt.close()


#print(df['sales'].unique())
# group by department of an employee
# and getting sum of all entries in one depart.
sales = df.groupby('sales').sum()
# print(sales)

# group by department of an employee
# and getting mean of all entries in one depart.
groupby_sales = df.groupby('sales').mean()
print(groupby_sales)

# mean satisfaction_level of each department
IT=groupby_sales['satisfaction_level'].IT
RandD=groupby_sales['satisfaction_level'].RandD
accounting=groupby_sales['satisfaction_level'].accounting
hr=groupby_sales['satisfaction_level'].hr
management=groupby_sales['satisfaction_level'].management
marketing=groupby_sales['satisfaction_level'].marketing
product_mng=groupby_sales['satisfaction_level'].product_mng
sales=groupby_sales['satisfaction_level'].sales
support=groupby_sales['satisfaction_level'].support
technical=groupby_sales['satisfaction_level'].technical

department_name = ('sales', 'accounting', 'hr', 'technical', 'support',
                   'management', 'IT', 'product_mng', 'marketing', 'RandD')

department = (sales, accounting, hr, technical, support, management,
              IT, product_mng, marketing, RandD)

y_pos = np.arange(len(department))
x = np.arange(0, 1, 0.1)

fig2 = plt.figure(2)
plt.barh(y_pos, department, align='center', alpha=0.8)
plt.yticks(y_pos, department_name)
plt.xlabel('Satisfaction Level')
plt.title('Mean Satisfaction Level of each department')
fig2.savefig('Mean Satisfaction Level of each department.png')
# plt.close()
# plt.show()

# Principal Component Analysis
print(df.head())

# dropping the columns in our dataset we don't need
df_drop = df.drop(labels=['sales', 'salary'], axis=1)
print(df_drop.head())

# to reshuffle the label column 'left' to the far left 
cols = df_drop.columns.tolist()
print(cols)

cols.insert(0, cols.pop(cols.index('left')))
print(cols)

df_drop = df_drop.reindex(columns=cols)
print(df_drop.head())

# separating features and label in our dataset
X = df_drop.iloc[:, 1:8].values
y = df_drop.iloc[:, 0].values
print(X)
print(y)
print(np.shape(X))
print(np.shape(y))
