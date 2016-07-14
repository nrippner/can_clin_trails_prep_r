import pandas as pd
from lxml import etree
import re
from os import listdir
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import pearsonr, spearmanr
import statsmodels.api as sm

path = 'C:/Users/Noah/Desktop/can_clin_trials/cancer/'
trials_files = listdir('%scancer_xml/' % path)
trial_names = []
for i, j in enumerate(trials_files):
    trial_names.append(j[:-4])

with open('%scancer_trials/study_fields.csv' % path) as f:
    df = pd.read_csv(f)
print "df <- pd.read_csv(study_fields.csv)"
print df.shape  
set(trial_names)==set(df['NCT Number'].values)
## True 


def get_xml(trialID):
    path = 'C:/Users/Noah/Desktop/can_clin_trials/cancer/'
    with open('%scancer_xml/%s.xml' % (path, trialID))  as f:
        parser = etree.XMLParser(ns_clean=True)          
        doc = etree.parse(f, parser)     
    return doc    
      
zips = {} 
def get_zips(trialID):
    global zips
    tree = get_xml(trialID)
    root = tree.getroot()
    temp = []
    for p in root:
        for c in p:
            for child in c.iterdescendants("zip"):
                temp.append(child.text)
            
    zip_codes=[]
    for i in temp:
        match = re.match(r'(\b)(\d{5})(\b)',i)
        if match:
            zip_codes.append(match.group())
    
    zips[trialID] = [zip_codes]
    
for trialID in trial_names:
    get_zips(trialID)
    
zipsdf = pd.DataFrame(zips)

# check for duplicates
zipsdf.shape[1] == len(set(zipsdf.columns))
    # True; therefore no duplicate columns

# Handle missing variables
    # count # of empty variables
missing_count = []
for i, j in enumerate(zipsdf.columns):
    if not zipsdf.loc[0,j]:
        missing_count.append(j)

print "# missing: ",len(missing_count)
    # # missing: 465
    
    # drop empty variables
for i in missing_count:
    zipsdf.drop(i, axis=1, inplace=True)

def strConvert(df, column):
    df.loc[column] = df[column].apply(lambda x: str(x))
    
def zeroes(data, col):
    for i, j in enumerate(data.loc[:,col]):
        if len(j) == 4:
            data.loc[i, col] = '0%s' % j
        elif len(j) == 3:
            data.loc[i, col] = '00%s' % j
        elif len(j) == 5:
            pass
        else:
            print i, j
    
            
zips_lists = []
for i in zipsdf.iloc[0,:]:
    zips_lists.append(i)

all_zips = [z for sublist in zips_lists for z in sublist]
print "total zips: ",len(all_zips)
print "unique zips: ", len(set(all_zips))
# total zips:  98564
# unique zips:  5297

countByZip = pd.Series(all_zips).value_counts()
countByZip = pd.DataFrame(countByZip,columns=['zipCount'])
countByZip.reset_index(level=0, inplace=True)
countByZip.rename(columns={'index':'zipCode'}, inplace=True)
strConvert(countByZip, 'zipCode')
print "countByZip: ", countByZip.shape
###############################################
## include zip codes with no clinical trials ##
###############################################
path = 'C:/Users/Noah/Desktop/can_clin_trials/zip_zcta/'
# downloaded from:
# https://www.census.gov/geo/maps-data/data/zcta_rel_download.html
with open('%scensus_zcta_fips.txt' % path) as f:
    fullZips = pd.read_csv(f)
print "fullZips <- pd.read_csv('census_zcta_fips.txt')"
print fullZips.shape # (44410, 24)
sum(fullZips['ZCTA5'].isnull())  # 0


columns = {'GEOID':'countyCode','ZCTA5':'zipCode'}
fullZips.rename(columns=columns, inplace=True)

fullZips.zipCode = fullZips.zipCode.astype('string')      
zeroes(fullZips, 'zipCode')
fullZips.countyCode = fullZips.countyCode.astype('string')      
zeroes(fullZips, 'countyCode')

countByZip.shape # (5298, 2)
isin = {}
icount = 0 # 4194
notin = {}
ncount = 0 # 1103

for i, j in enumerate(countByZip.zipCode):
    if fullZips.zipCode.apply(lambda x: x == j).any():
        isin[i] = j
        icount += 1
    elif not fullZips.zipCode.apply(lambda x: x == j).any():
        notin[i] = j
        ncount += 1

fullZips = fullZips[fullZips.ZPOPPCT > 50.0]

print "# of zip codes in countByZip that are in fullZips: ", icount
print "# not in fullZips: ", ncount
# of zip codes in countByZip that are in fullZips:  3749
# not in fullZips:  1549



#### here #####




fullZips = pd.merge(fullZips, countByZip, how='left', on='zipCode')
print "fullZips <- pd.merge(fullZips, countByZip, how='left', on='zipCode')"
len(fullZips) - sum(fullZips.zipCount.isnull()) # 4609
fullZips.zipCount.fillna(0, inplace=True)
fullZips.rename(columns={'zipCount':'studyCount'}, inplace=True)
fullZips.shape  # (43582, 7)

d = len([i for i in fullZips.zipCode.duplicated() if i])
print "fullZips:"
print "# unique zips: ", len(fullZips.zipCode.unique())
print "# of duplicate zips: %d" % d
print "# zips with > 0 clin trials: ", \
                    sum(fullZips.studyCount.apply(lambda x: x > 0))
print "# zips with 0 clin trials: ", \
                    sum(fullZips.studyCount.apply(lambda x: x == 0))
print fullZips.columns
# of duplicate zips: 0
# zips with > 0 clin trials:  3719
# zips with 0 clin trials:  29143


####################################
## Census data: income by county ##
###################################
# downloaded from census.gov
path = 'c:/Users/Noah/Desktop/can_clin_trials/'
with open('%scensus_data/cen_income.csv' % path) as f:
    cdf = pd.read_csv(f)
print "cdf <- pd.read_csv('census_income.csv')"
print cdf.shape
columns = {'State FIPS Code':'SFIPS','County FIPS Code':'CFIPS',
           'Postal Code':'State','Poverty Percent, All Ages':'PovertyPercent',
           'Poverty Estimate, All Ages':'PovertyEst',
           'Median Household Income':'medIncome'}
cdf.rename(columns=columns, inplace=True)



cdf.SFIPS = cdf.SFIPS.astype('string',inplace=True)
cdf.CFIPS = cdf.CFIPS.astype('string',inplace=True)

for i, j in enumerate(cdf['SFIPS']):
    if len(j) == 1:
        cdf.loc[i, 'SFIPS'] = '0%s' % j

for i, j in enumerate(cdf['CFIPS']):
    if len(j) == 1:
        cdf.loc[i, 'CFIPS'] = '00%s' % j
    elif len(j) == 2:
        cdf.loc[i, 'CFIPS'] = '0%s' % j
        
cdf['countyCode'] = np.nan
for i, j in enumerate(cdf['CFIPS']):
    cdf.loc[i, 'countyCode'] = '%s%s' % (cdf.loc[i,'SFIPS'],
                                       cdf.loc[i,'CFIPS'])

cdf = cdf[['State','PovertyEst','PovertyPercent','medIncome','countyCode',
                                                                    'Name']]
print "cdf = cdf[['State','PovertyEst','PovertyPercent','medIncome',\
                                    'countyCode','Name']]"
print "cdf: \n", cdf.shape  # (3194, 6)
d = len([i for i in cdf.countyCode.duplicated() if i])
print "# unique counties: ", len(cdf.countyCode.unique())
print "# of duplicate counties: %d" % d
print cdf.columns

# population by county
with open('census_county_population.txt') as f:
    popdf = pd.read_csv(f)
# downloaded from 
# https://www.census.gov/popest/data/counties/totals/2015/index.html
print "popdf <- pd.read_csv('census_county_population.txt')"
print popdf.shape
popdf.rename(columns={'STATE':'SFIPS','COUNTY':'CFIPS'},inplace=True)

popdf.SFIPS = popdf.SFIPS.astype('string',inplace=True)
popdf.CFIPS = popdf.CFIPS.astype('string',inplace=True)

for i, j in enumerate(popdf['SFIPS']):
    if len(j) == 1:
        popdf.loc[i, 'SFIPS'] = '0%s' % j

for i, j in enumerate(popdf['CFIPS']):
    if len(j) == 1:
        popdf.loc[i, 'CFIPS'] = '00%s' % j
    elif len(j) == 2:
        popdf.loc[i, 'CFIPS'] = '0%s' % j
        
popdf['countyCode'] = np.nan
for i, j in enumerate(popdf['CFIPS']):
    popdf.loc[i, 'countyCode'] = '%s%s' % (popdf.loc[i,'SFIPS'],
                                       popdf.loc[i,'CFIPS'])
popdf.drop(['SFIPS','CFIPS'], inplace=True, axis=1)
print "popdf: "
print popdf.shape
d = len([i for i in popdf.countyCode.duplicated() if i])
print "# of duplicate counties: %d" % d
print "# of unique counties: ", len(popdf.countyCode.unique())
print popdf.columns

cdf = pd.merge(cdf, popdf[['countyCode', 'POPESTIMATE2015']], how='inner',
                on='countyCode')
print "cdf <- pd.merge(cdf, popdf[['countyCode', 'POPESTIMATE2015']], \
                how='inner', on='countyCode')"
print cdf.shape # (3191, 7)  
print cdf.columns



  
data = pd.merge(fullZips, cdf, how='inner', on='countyCode')
print "data <- pd.merge(data, cdf, how='inner', on='countyCode')"


data.medIncome = data.medIncome.apply(lambda x: re.sub(r',', '', x)) 
data.PovertyEst = data.PovertyEst.apply(lambda x: re.sub(r',', '', x))

for i, j in enumerate(data.PovertyEst):
    try:
        data.loc[i,'PovertyEst'] = float(j)
    except:
        data.loc[i,'PovertyEst'] = float(0)

for i, j in enumerate(data.PovertyEst):
    if j == 0:
        print i,j,np.mean(data.PovertyEst)
        data.loc[i,'PovertyEst'] = np.mean(data.PovertyEst)

data.PovertyEst = data.PovertyEst.astype('float64')

for i, j in enumerate(data.PovertyPercent):
    try:
        data.loc[i, 'PovertyPercent'] = float(j)
    except:
        data.loc[i, 'PovertyPercent'] = float(0)
        print i, j
# 31479 0.0 67019.2550135
# 31479 .

data.drop(31479, inplace=True)

print "data:"
print data.shape 
print "# of unique zips: ", len(data.zipCode.unique())
print "# of unique counties: ", len(data.countyCode.unique())
d = len([i for i in data.zipCode.duplicated() if i])
print "# of duplicate zips: %d" % d
d = len([i for i in data.countyCode.duplicated() if i])
print "# of duplicate counties: %d" % d
print data.columns

data = data[['zipCode','countyCode','studyCount','State','PovertyEst',
             'PovertyPercent','medIncome','Name','POPESTIMATE2015']]







## merge cancer incidence rates by county
path = 'C:/Users/Noah/Desktop/can_clin_trials/cancer/incidence/'
with open('%sincd_r.csv' % path) as f:
    incdf = pd.read_csv(f)
# downloaded from:
# http://statecancerprofiles.cancer.gov/incidencerates/index.php?stateFIPS=
#        51&cancer=071&race=00&sex=0&age=001&type=incd#results
print "incdf <- pd.read_csv('incdr.csv')"
print "incdf:"
print incdf.shape 
print "# of unique counties: ", len(data.countyCode.unique())
print incdf.columns


sum(incdf.countyCode.isnull()) # 0
incdf.dropna(subset=['countyCode'], inplace=True)
incdf.countyCode = incdf.countyCode.apply(lambda x: int(x))
incdf.countyCode=incdf.countyCode.astype('string')
zeroes(incdf, 'countyCode')
incdf.rename(columns={'5YearTrend':'fiveYearTrend'}, inplace=True)

print "data:\n# of unique counties: ", len(data.countyCode.unique())
data.shape 

data = pd.merge(data, incdf, how='inner', on='countyCode')
print "data <- pd.merge(data, incdf, how='inner', on='countyCode')"
print "data:\n# of unique counties: ", len(data.countyCode.unique())
print data.shape 
print data.columns



## cancer death rates by county
# https://statecancerprofiles.cancer.gov/cgi-bin/deathrates/
#    deathrates.pl?99&001&00&0&001&0&1&1&1#results
path = 'C:/Users/Noah/Desktop/can_clin_trials/cancer/'
with open('%sdeath_r.csv' % path) as f:
    mort = pd.read_csv(f)
# (3157, 11)
# rename columns
columns = {}
names = ['countyName','countyCode','metObj','deathRate','l95DR','u95DR',
         'avgDeathsPerYear','recTrend','fiveYearTrend','l95T','u95T']
for old, new in zip(mort.columns, names):
    columns[old] = new

mort.rename(columns=columns, inplace=True)

# add leading zeros to countyCode

mort.dropna(inplace=True)
mort.countyCode = mort.countyCode.astype('int64')
mort = mort[mort.countyCode > 0]
mort.reset_index(drop=True, inplace=True)
mort.countyCode = mort.countyCode.astype('string')
zeroes(mort, 'countyCode')
mort = mort[['countyName','countyCode','deathRate','avgDeathsPerYear',
             'recTrend']]

mort.avgDeathsPerYear = mort.avgDeathsPerYear.str.replace(',','')

d = {}
for i in ['deathRate','avgDeathsPerYear']:
    for ii, j in enumerate(mort[i]):
        try:
            mort.loc[ii,i] = float(j)
        except:
            try:
                d[ii].append((i,j))
            except:
                d[ii] = [(i,j)]
print "# counties with no reported deathRate data: ", len(d)
# counties with no reported deathRate data:  55

mort.drop(d.keys(), inplace=True)

mort.deathRate = mort.deathRate.astype('float64')
mort.avgDeathsPerYear = mort.avgDeathsPerYear.astype('float64')


print "mort <- pd.read_csv('death_r.csv')"
print "mort:"
print mort.shape
print "# of unique counties: ", len(mort.countyCode.unique())
print mort.columns


data = pd.merge(data, mort, how='inner', on='countyCode')
print "data <- pd.merge(data, mort, how='inner', on='countyCode')"
print "data:\n# of unique counties: ", len(data.countyCode.unique())
print data.shape # 
print data.columns


# convert features to float
data.rename(columns={'POPESTIMATE2015':'popEst2015','PovertyPercent':
                        'povertyPercent'}, inplace=True)
for i in ['incidenceRate','medIncome','avgAnnCount','popEst2015']:
    data[i] = data[i].astype('string')

for i, j in enumerate(data.incidenceRate):
    try:
        data.loc[i,'incidenceRate'] = re.findall("\d+\.\d+|\d+", j)[0]
    except:
        data.loc[i,'incidenceRate'] = np.nan
data.incidenceRate = data.incidenceRate.astype('float64')

data.medIncome = data.medIncome.astype('float64')

for i, j in enumerate(data.avgAnnCount):
    try:
        data.loc[i,'avgAnnCount'] = float(j)
    except:
        data.loc[i,'avgAnnCount'] = np.nan



data.popEst2015 = data.popEst2015.astype('float64')
data.povertyPercent = data.povertyPercent.astype('float64') 

print 'data:\nincidenceRate # nan: ', sum(data.incidenceRate.isnull())
print 'medIncome # nan: ', sum(data.medIncome.isnull())
print 'avgAnnCount # nan: ', sum(data.avgAnnCount.isnull())
print 'popEst2015 # nan: ', sum(data.popEst2015.isnull())


data.incidenceRate.fillna(value=data.incidenceRate.mean(), inplace=True)
data.avgAnnCount.fillna(value=data.avgAnnCount.mean(), inplace=True)

dz = len([i for i in data.zipCode.duplicated() if i])
dc = len([i for i in data.countyCode.duplicated() if i])
print "data:"
print data.shape
print "# of duplicate zips: %d" % dz
print "# of duplicate counties: %d" % dc
print "# unique zips: ", len(data.zipCode.unique())
print "# unique counties: ", len(data.countyCode.unique())
print data.columns



# group by county & zip code
countyData = pd.pivot_table(data, index=['countyCode'])
countyData.reset_index(inplace=True)
t = pd.pivot_table(data, index=['countyCode'], aggfunc='sum')
countyData.drop('studyCount', axis=1, inplace=True)
countyData = countyData.assign(studyCount = [i for i in t.studyCount])
countyData['studyPerCap'] = (countyData.studyCount / 
                                    countyData.popEst2015 * 10**6)
# studies per 1,000,000 people 
zipCodeData = pd.pivot_table(data, index=['zipCode'])
zipCodeData.reset_index(inplace=True)
zipCodeData['studyPerCap'] = (zipCodeData.studyCount /
                                zipCodeData.popEst2015 * 10**6)
# studies per 1,000,000 people 

d = len([i for i in zipCodeData.zipCode.duplicated() if i])
print "zipCodeData:"
print zipCodeData.shape
print "# of duplicate zips: %d" % d
print "# unique zips: ", len(zipCodeData.zipCode.unique())
print zipCodeData.columns


print "countyData:"
print countyData.shape
print countyData.shape
d = len([i for i in countyData.countyCode.duplicated() if i])
print "# of duplicate counties: %d" % d
print "# unique counties: ", len(countyData.countyCode.unique())
print countyData.columns

# discretize medIncome
quantiles = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]

    # countyData
countyCuts = countyData['medIncome'].quantile(quantiles).values

countyBinnedInc = pd.cut(countyData.medIncome, bins=countyCuts, right=True, 
                   include_lowest=True)
countyBinnedInc.value_counts()
countyData['binnedInc'] = countyBinnedInc

countyBinned = countyData.groupby('binnedInc').mean()

countyBinned = countyBinned.reset_index()

    # zipCodeData
zipCuts = zipCodeData['medIncome'].quantile(quantiles).values

zipBinnedInc = pd.cut(zipCodeData.medIncome, bins=zipCuts, right=True, 
                   include_lowest=True)
zipBinnedInc.value_counts()
zipCodeData['binnedInc'] = countyBinnedInc

zipBinned = zipCodeData.groupby('binnedInc').mean()

zipBinned = zipBinned.reset_index()


######################################################################3

y = np.asarray(countyData.studyPerCap)
X1 = countyData.copy()
X1.drop(['studyCount','binnedInc','countyCode','studyPerCap'],axis=1,inplace=True)
X = np.asmatrix(X1)
X = sm.add_constant(X)

mod = sm.OLS(y, X)
res = mod.fit()
print res.summary()










cancer = []
cancerLabels = []
bInc   = []
popEst = []
   
   
for i, j in enumerate(countyData.deathRate):
    cancer.append(j)
    cancerLabels.append("Death Rate")
    bInc.append(countyData.loc[i, 'binnedInc'])
    popEst.append(countyData.loc[i,'popEst2015'])

for i, j in enumerate(countyData.incidenceRate):
    cancer.append(j)
    cancerLabels.append("Incidence Rate")
    bInc.append(countyData.loc[i, 'binnedInc'])
    popEst.append(countyData.loc[i,'popEst2015'])
    
            
iddf = pd.DataFrame({'cancer':cancer,'cancerLabels':cancerLabels,'bInc':bInc,
                     'popEst':popEst})

order = ['[22640, 34017.5]', '(34017.5, 37151]', '(37151, 40121]', 
         '(40121, 42302]', '(42302, 44822.5]', '(44822.5, 47493]',
         '(47493, 50706]', '(50706, 54086]', '(54086, 60447.5]',
         '(60447.5, 125635]']                                        
bar = sns.barplot(x="bInc", y="cancer", hue="cancerLabels", order=order, 
                  data=iddf)
bar.set_title(label=
        "Cancer Incidence Rate by Median Household Income:\nBy County(decile)")
plt.show()


bar = sns.barplot(x="binnedInc", y="deathRate", data=countyData)
bar.set_xticklabels(labels=range(1,11))
bar.set_title(label=
              "Cancer Death Rate by Median Household Income By County\
               \nAll Cancers -- Deaths per 100,000")
bar.set_xlabel(xlabel="Income Decile")
bar.set_ylabel(ylabel="Death Rate") 
plt.show()

bar = sns.barplot(x="binnedInc", y="incidenceRate", data=countyData)
bar.set_xticklabels(labels=range(1,11))
bar.set_title(label=
              "Cancer Incidence Rate by Median Household Income By County\
               \nAll Cancers -- Rates per 100,000")
bar.set_xlabel(xlabel="Income Decile")
bar.set_ylabel(ylabel="Incidence Rate") 
plt.show()

bar = sns.barplot(x="binnedInc", y="studyPerCap", data=countyData)
bar.set_xticklabels(labels=range(1,11))
bar.set_title(label=
              "Cancer Incidence Rate by Median Household Income By County\
               \nAll Cancers -- Rates per 100,000")
bar.set_xlabel(xlabel="Income Decile")
bar.set_ylabel(ylabel="Incidence Rate") 
plt.show()



countyBinned[['binnedInc','deathRate','incidenceRate']]


      
            
                  
                              
bar = sns.barplot(x="binnedInc", y="studyCount", data=countyData)
bar.set_xticklabels(labels=range(1,11))
bar.set_title(label= "Study Count by Median Household Income By County")
bar.set_xlabel(xlabel="Income Decile")
bar.set_ylabel(ylabel="Study Count") 
plt.show()


bar = sns.barplot(x="binnedInc", y="popEst2015", data=countyData)
bar.set_xticklabels(labels=range(1,11))
bar.set_title(label= "Population by Median Household Income By County")
bar.set_xlabel(xlabel="Income Decile")
bar.set_ylabel(ylabel="Population") 
plt.show()

bar = sns.barplot(x="binnedInc", y="popEst2015", data=countyData)
bar.set_xticklabels(labels=range(1,11))
bar.set_title(label= "Population by Median Household Income By County")
bar.set_xlabel(xlabel="Income Decile")
bar.set_ylabel(ylabel="Population") 
plt.show()


sns.set(color_codes=True)
sns.regplot(x="popEst2015", y="deathRate", data=countyData, lowess=False)
#sns.regplot(x='medIncome', y='deathRate', data=countyData, lowess=False)

sns.set(color_codes=True)
sns.regplot(x="popEst2015", y="studyCount", data=countyData, lowess=False)




