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
print('Columns names : ')
print(columns_names)
##print(df.shape)
##print(df.head())
print(df.corr())

# visualising correlation using seaborn library
correlation = df.corr()
plt.figure(figsize=(10, 10))
ax = sns.heatmap(correlation, vmax=1, square=True, annot=True, cmap='cubehelix')
bottom, top = ax.get_ylim()
ax.set_ylim(bottom + 0.5, top - 0.5)

plt.title('Correlation between different features')
#plt.show()


print(df['sales'].unique())
