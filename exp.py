import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
# Lichman, M. (2013). UCI Machine Learning Repository [http://archive.ics.uci.edu/ml]. Irvine, CA: University of California, School of Information and Computer Science.

path = 'C:/Users/Noah/Desktop/open_data/credit_approval/credit.txt'
with open(path) as f:
    df = pd.read_csv(f)

# data "tidying"
df.columns = range(len(df.columns))

for i, j in enumerate(df[1]):
    try:
        df.loc[i,1] = float(j)
    except:
        df.loc[i,1] = np.nan

df.loc[:,1] = df.loc[:,1].astype('float64')

num = [1,2,7,14]
cat = [0,3,4,5,6,8,9,10,11,12,13,15]
for i in cat:
    df.loc[:,i] = df[i].astype('category')

numd = {}
for i, j in enumerate(num):
    numd[j] = 'Num%s' % i
for i, j in enumerate(cat):
    numd[j] = 'Cat%s' % i

df.columns = numd.values()


# add quantiles, skew, kurt to df.describe()
nsdf = df.describe(percentiles=[.01,.05,.10,.25,.50,.75,.90,.95,.99])
skewdf = df.skew().to_frame().transpose().rename(index={0:'skew'})
kurtdf = df.kurt().to_frame().transpose().rename(index={0:'kurt'})
nsdf = nsdf.append([skewdf, kurtdf]).round(2)

# frequency table categorical variables
freq_table = pd.crosstab(index=df[4], columns=[df['Cat5'], df['Cat6']], 
             margins=True)

# contingency table categorical and continuous
table = df.pivot_table(['Num2', 'Num3'], index=['Cat0', 'Cat5'], aggfunc=np.mean)

# seaborn stripplot w/ jitter
ax = sns.stripplot(x='Cat1', y='Num0', data=df, jitter=True, hue='Cat0')

# boxplots
df.boxplot(column=['Num0'], by=['Cat1','Cat0'])

# histogram w/ rug plot
ax = sns.distplot(df['Num0'][df['Num0']>0], rug=True, bins=50)
ax.set_title(label='Histogram with rug & KDE')
plt.show()

# histogram matrix
df.hist(bins=50)

# scatterplot matrix
sns.pairplot(df, vars=['Num0','Num1','Num2'], hue='Cat4', dropna=True)

# correlation matrix
df.corr()


df.to_csv('credit.csv')
nsdf.to_csv('desc.csv')