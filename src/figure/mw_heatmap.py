# %% Packages
import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt

file_path = '../../res/eye_features_whole.csv'
df = pd.read_csv(file_path)

# %% Group DataFrame
df_group = df.groupby(['reading', 'page_num'])['reported_MW'].agg(['sum', 'count']).assign(ratio=lambda x: x['sum'] / x['count']).reset_index()

df_group.rename(columns={'ratio':'MW Percentage'}, inplace=True)


# %%
data = df_group.pivot(index="reading", columns="page_num", values="MW Percentage")

# Plot the heatmap
plt.figure(figsize=(12, 5))
sns.heatmap(data, annot=True, fmt=".1%", cmap="YlGnBu", cbar_kws={'label': 'MW Percentage (%)'})
plt.title("Heatmap of Mind-wandering Percentage Across Stories and Pages")
plt.xlabel("Page Number")
plt.ylabel("Stories")
plt.tight_layout()
plt.show()