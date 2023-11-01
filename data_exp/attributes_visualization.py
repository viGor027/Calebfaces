import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from src.constants import CSV_PATH

attr = pd.read_csv(os.path.join(CSV_PATH, 'list_attr_celeba.csv'))

sums = attr.sum(numeric_only=True, axis=0).reset_index()
sums[0] = np.abs(sums[0])
sums.sort_values(by=0, axis=0, inplace=True)

plt.figure(figsize=(19, 13))
sns.set(font_scale=1.6)
chart = sns.barplot(sums.iloc[-10:], x=sums['index'].iloc[-10:], y=sums[0].iloc[-10:], width=0.6)
chart.set_title('Attributes_balance')
chart.set_ylabel('Abs_of_column_sum')
plt.xlabel('Attribute')
plt.xticks(rotation=30)
plt.savefig('10_most_imbalanced_attributes')
