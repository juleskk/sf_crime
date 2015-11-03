from dateutil import parser
import datetime
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns; sns.set(color_codes=True)
%matplotlib inline

df = pd.read_csv('...', parse_dates=['dates'])
df['category'] = df['category'].apply(lambda x: x.title())
dummies = pd.get_dummies(df['hour'])
dummies['category'] = df['category']
gp = dummies.groupby('category').sum()
gp.sort(ascending=False, inplace=True)

fig = plt.figure(figsize=(16, 12))
ax = fig.add_subplot(111)

sns.heatmap(gp, linewidths=.5, cmap= "GnBu")
