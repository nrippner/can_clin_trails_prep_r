import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

path = 'C:/Users/Noah/Desktop/open_data/refugee_refs/UN/'
with open('%sunhcr_2015.csv' % path) as f:
    df_2015 = pd.read_csv(f)

with open('%sunhcr_all.csv' % path) as f:
    df_all = pd.read_csv(f)


df_all = df_all.append(df_2015)
df_all.to_csv('%srefugees_all_years.csv' % path)


df = df_all[df_all.Year > 2010]
df = df.iloc[:,[0,1,2,3,4,-1]]
df.columns = ['Year','Country','Origin','Refugees','AsylumSeekers','TotalPop']
df.reset_index(drop=True, inplace=True)

for col in ['Refugees', 'AsylumSeekers', 'TotalPop']:
    for i, j in enumerate(df.loc[:,col]):
        try:
            df.loc[i, col] = float(j)
        except:
            df.loc[i, col] = np.nan

df['Origin'] = df.Origin.str.replace('^Venezuela.*','Venezuela')
df['Origin'] = df.Origin.str.replace('^Russian.*','Russia')
df['Origin'] = df.Origin.str.replace('^United States.*','UnitedStates')
df['Origin'] = df.Origin.str.replace('^Iran.*','Iran')
df['Origin'] = df.Origin.str.replace('^.*Congo.*$','Congo')
df['Country'] = df.Country.str.replace('^Venezuela.*','Venezuela')
df['Country'] = df.Country.str.replace('^Russian.*','Russia')
df['Country'] = df.Country.str.replace('^United States.*','UnitedStates')
df['Country'] = df.Country.str.replace('^Iran.*','Iran')
df['Country'] = df.Country.str.replace('^.*Congo.*$','Congo')

df.to_csv('%srefugees2011-15.csv' % path, index=False)

cdf = df.set_index(['Country','Year']).sort_values(by=['Refugees'], 
                                                            ascending=False)
cdf = cdf.sort_index()



    
r = range(2011,2016)
c = ['darkgoldenrod','moccasin','darkgrey','lightgreen','violet']    
n = 0
for i in range(5):
    cdf.loc['US'].loc[r[i]].head(10).plot(x='Origin',y='Refugees',kind='bar',
                color=c[i])
    plt.title(s='United States Refugees Accepted by Origin Country\n%d' % r[i])
    plt.xticks(rotation=60)
    plt.subplots_adjust(bottom=.2)
    plt.ylabel("Number of Immigrants Accepted")
    plt.xlabel("Origin Country")
    plt.show()
    n += 1

df[(df.Refugees.notnull()) & (df.Year == 2015)].sort_values(by='Refugees', 
                                                    ascending=False).head(10)
                                                    
i = -3
data = df[(df.Refugees.notnull()) & (df.Year == r[i])]
data = data.sort_values(by='Refugees', ascending=False).head(15)
bar = sns.barplot(y="Country", x="Refugees", orient="h" , data=data,
                    palette="dark", estimator=sum)

bar.set_title(label=
            "Nations Accepting Highest Numbers of Refugees\n%s" % r[i])
bar.set_xlabel(xlabel="Refugees")
bar.set_ylabel(ylabel="Host Country") 
plt.subplots_adjust(left=.2)
plt.show()

