# -*- coding: utf-8 -*-
"""
Created on Fri Nov  6 19:18:43 2020

@author: xinli
"""
import os 
import json
import requests
import pandas as pd
from matplotlib import pyplot as plt
from scipy.stats import gamma
import numpy as np
import pycurl
import re
from matplotlib.cm import ScalarMappable
from mpl_toolkits.basemap import Basemap as Basemap
from matplotlib.colors import rgb2hex,Normalize
from matplotlib.patches import Polygon
from matplotlib.colorbar import ColorbarBase
from geopy.geocoders import Nominatim
import math

os.getcwd() 
os.chdir('C:\\Users\\xinli\\Downloads\\c3aidatalake-notebooks-python\\c3aidatalake-notebooks-python') 
os.getcwd() 
import c3aidatalake
os.chdir('C:\\Users\\xinli')
print("pandas version", pd.__version__)
assert pd.__version__[0] >= "1", "To use this notebook, upgrade to the newest version of pandas. See https://pandas.pydata.org/pandas-docs/stable/getting_started/install.html for details."

states = [
  'Alabama_UnitedStates','Alaska_UnitedStates','Arizona_UnitedStates',
  'Arkansas_UnitedStates','California_UnitedStates','Colorado_UnitedStates',
  'Connecticut_UnitedStates','Delaware_UnitedStates','DistrictofColumbia_UnitedStates',
  'Florida_UnitedStates','Georgia_UnitedStates','Hawaii_UnitedStates',
  'Idaho_UnitedStates','Illinois_UnitedStates','Indiana_UnitedStates',
  'Iowa_UnitedStates','Kansas_UnitedStates','Kentucky_UnitedStates',
  'Louisiana_UnitedStates','Maine_UnitedStates','Maryland_UnitedStates',
  'Massachusetts_UnitedStates','Michigan_UnitedStates','Minnesota_UnitedStates',
  'Mississippi_UnitedStates','Missouri_UnitedStates','Montana_UnitedStates',
  'Nebraska_UnitedStates','Nevada_UnitedStates','NewHampshire_UnitedStates',
  'NewJersey_UnitedStates','NewMexico_UnitedStates','NewYork_UnitedStates',
  'NorthCarolina_UnitedStates','NorthDakota_UnitedStates','Ohio_UnitedStates',
  'Oklahoma_UnitedStates','Oregon_UnitedStates','Pennsylvania_UnitedStates',
  'PuertoRico_UnitedStates','RhodeIsland_UnitedStates','SouthCarolina_UnitedStates',
  'SouthDakota_UnitedStates','Tennessee_UnitedStates','Texas_UnitedStates',
  'Utah_UnitedStates','Vermont_UnitedStates','Virginia_UnitedStates',
  'Washington_UnitedStates','WestVirginia_UnitedStates','Wisconsin_UnitedStates',
  'Wyoming_UnitedStates']

# Fetch facts about Germany
locations = c3aidatalake.fetch(
    "outbreaklocation",
    {
        "spec" : {
            "filter" : "id == 'California_UnitedStates'"
        }
    }
)

# Fetch participants who are located in California and who have a relatively strong intent to wear a mask in public because of COVID-19
survey = c3aidatalake.fetch(
    "surveydata",
    {
        "spec": {
            "filter": "location == 'California_UnitedStates' && coronavirusIntent_Mask >= 75"
        }
    },
    get_all = True
)

employment_df = survey.copy()
employment_df["coronavirusEmployment"] = employment_df["coronavirusEmployment"].str.split(", ")
employment_df = employment_df.explode("coronavirusEmployment")
employment_df = employment_df.groupby(["coronavirusEmployment"]).agg("count")[["id"]].sort_values("id")

# Plot the data
plt.figure(figsize = (10, 6))
plt.bar(employment_df.index, 100 * employment_df["id"] / len(survey))
plt.xticks(rotation = 90)
plt.xlabel("Response to employment status question")
plt.ylabel("Proportion of participants (%)")
plt.title("Employment status of CA participants with strong intent to wear mask")
plt.show()

def create_headers(bearer_token):
    headers = {"Authorization": "Bearer {}".format(bearer_token)}
    return headers


def get_rules(headers, bearer_token):
    response = requests.get(
        "https://api.twitter.com/2/tweets/search/stream/rules", headers=headers
    )
    if response.status_code != 200:
        raise Exception(
            "Cannot get rules (HTTP {}): {}".format(response.status_code, response.text)
        )
    print(json.dumps(response.json()))
    return response.json()

headers = {
    'authorization': 'Bearer AAAAAAAAAAAAAAAAAAAAABkLJgEAAAAA1xxKTYI%2F83ePfIVUwzozbsGbbQk%3D0tES8qrsNGcOzxRaqkrk98vHcVBJB4Sp8gtsEvbNcyY3Agvz7k',
}

params = (
    ('ids', '1278747501642657792'),
)

response = requests.get('https://api.twitter.com/2/tweets', headers=headers, params=params)

import tweepy as tw
consumer_key= 'QRbuCzE4GszQY572KxrnrIRTN'
consumer_secret= '7kjGXOnyi3D9jaEvyXuMyQ8oAGeH7ZqrMJX37ztSxF2zN9Udzh'
access_token= '1613811230-dlOvFagU4KQi0bHkML0Ymltqjwwhd9oV4kX1InB'
access_token_secret= 'mPIMQ97ItwRaeKkgZCUSyegkQ8ra9Eo2FG6BJ0eCTBUiU'


auth = tw.OAuthHandler(consumer_key, consumer_secret)
auth.set_access_token(access_token, access_token_secret)
api = tw.API(auth, wait_on_rate_limit=True)

date_since = "2020-10-08"
new_search="#flattenthecurve"
tweets = tw.Cursor(api.search, 
                           q=new_search,
                           lang="en",
                           since=date_since).items(100)
# Iterate and print tweets
i=0
df = pd.DataFrame(columns=['Name', 'Location', 'Time'])
for tweet in tweets:
    df.loc[i]= [tweet.user.screen_name, tweet.user.location, tweet.created_at]
    i=i+1

#replace empty locations with nan and remove them
df['Location'].replace('', np.nan, inplace=True)
df.dropna(subset=['Location'], inplace=True)
df.to_csv('flattenthecurve.csv')
count = df['Location'].value_counts()

date_since = "2020-10-31"
new_search="#stayhome"
tweets = tw.Cursor(api.search, 
                           q=new_search,
                           lang="en",
                           since=date_since).items(100)
# Iterate and print tweets
i=0
df = pd.DataFrame(columns=['Name', 'Location', 'Time'])
for tweet in tweets:
    df.loc[i]= [tweet.user.screen_name, tweet.user.location, tweet.created_at]
    i=i+1

#replace empty locations with nan and remove them
df['Location'].replace('', np.nan, inplace=True)
df.dropna(subset=['Location'], inplace=True)
df.to_csv('stayhome.csv')
count = df['Location'].value_counts()


survey = c3aidatalake.fetch(
    "surveydata",
    {
        "spec": {
            
        }
    },
    get_all = True
)

survey['startTime'] = pd.to_datetime(survey['startTime'])
survey['startTime'] = survey['startTime'].apply(lambda t: t.replace(second=0))
survey['startTime'] = survey['startTime'].apply(lambda t: t.replace(minute=0))
survey['startTime'] = survey['startTime'].apply(lambda t: t.replace(hour=0))
surveydates= survey['startTime'].value_counts()

# importing datetime module 
from datetime import *
d = date(2020, 6, 1) 
today = date.today()

rslt_df = survey.loc[survey['startTime'] > '2020-06-06'] 
rslt_df2 = survey.loc[survey['startTime'] < '2020-05-16'] 
rslt_df2["location.id"] = rslt_df2["location.id"].str.replace("_UnitedStates", "")
#s.environ["PROJ_LIB"] = "C:\\Users\\xinli\\Anaconda3\\Library\\share"; #fixr

may2 = rslt_df2[["location.id","coronavirusIntent_Mask"]]
count = may2['location.id'].value_counts()

may2 = may2[may2['coronavirusIntent_Mask'].notna()]
may2["location.id"] = may2["location.id"].str.replace("_UnitedStates", "")
may2["location.id"] = may2["location.id"].str.replace("SouthCarolina", "South Carolina")
may2["location.id"] = may2["location.id"].str.replace("NorthCarolina", "North Carolina")
may2["location.id"] = may2["location.id"].str.replace("DistrictofColumbia", "District of Columbia")
may2["location.id"] = may2["location.id"].str.replace("NewHampshire", "New Hampshire")
may2["location.id"] = may2["location.id"].str.replace("SouthDakota", "South Dakota")
may2["location.id"] = may2["location.id"].str.replace("NorthDakota", "North Dakota")
may2["location.id"] = may2["location.id"].str.replace("NewYork", "New York")
may2["location.id"] = may2["location.id"].str.replace("NewMexico", "New Mexico")
may2["location.id"] = may2["location.id"].str.replace("PuertoRico", "Puerto Rico")
may2["location.id"] = may2["location.id"].str.replace("WestVirginia", "West Virginia")
may2["location.id"] = may2["location.id"].str.replace("NewJersey", "New Jersey")
may2["location.id"] = may2["location.id"].str.replace("RhodeIsland", "Rhode Island")

may4=pd.DataFrame(may2.groupby('location.id')['coronavirusIntent_Mask'].mean())
dictionary = may4['coronavirusIntent_Mask'].to_dict()

june2 = rslt_df[["location.id","coronavirusIntent_Mask"]]
count = june2['location.id'].value_counts()

june2 = june2[june2['coronavirusIntent_Mask'].notna()]
june2["location.id"] = june2["location.id"].str.replace("_UnitedStates", "")
june2["location.id"] = june2["location.id"].str.replace("SouthCarolina", "South Carolina")
june2["location.id"] = june2["location.id"].str.replace("NorthCarolina", "North Carolina")
june2["location.id"] = june2["location.id"].str.replace("DistrictofColumbia", "District of Columbia")
june2["location.id"] = june2["location.id"].str.replace("NewHampshire", "New Hampshire")
june2["location.id"] = june2["location.id"].str.replace("SouthDakota", "South Dakota")
june2["location.id"] = june2["location.id"].str.replace("NorthDakota", "North Dakota")
june2["location.id"] = june2["location.id"].str.replace("NewYork", "New York")
june2["location.id"] = june2["location.id"].str.replace("NewMexico", "New Mexico")
june2["location.id"] = june2["location.id"].str.replace("PuertoRico", "Puerto Rico")
june2["location.id"] = june2["location.id"].str.replace("WestVirginia", "West Virginia")
june2["location.id"] = june2["location.id"].str.replace("NewJersey", "New Jersey")
june2["location.id"] = june2["location.id"].str.replace("RhodeIsland", "Rhode Island")

june4=pd.DataFrame(june2.groupby('location.id')['coronavirusIntent_Mask'].mean())
junesurvey=pd.DataFrame(june2.groupby('location.id',as_index=False)['coronavirusIntent_Mask'].mean())
dictionary2 = june4['coronavirusIntent_Mask'].to_dict()
# Lambert Conformal map of lower 48 states.
m = Basemap(llcrnrlon=-119,llcrnrlat=22,urcrnrlon=-64,urcrnrlat=49,
        projection='lcc',lat_1=33,lat_2=45,lon_0=-95)
shp_info = m.readshapefile('C:\\Users\\xinli\\Downloads\\st99_d00','states',drawbounds=True)

colors={}
statenames=[]
cmap = plt.cm.hot # use 'hot' colormap
vmin = 0; vmax = 450 # set range.
for shapedict in m.states_info:
    statename = shapedict['NAME']
    # skip DC and Puerto Rico.
    if statename not in ['District of Columbia','Puerto Rico']:
        pop = dictionary[statename]
        # calling colormap with value between 0 and 1 returns
        # rgba value.  Invert color range (hot colors are high
        # population), take sqrt root to spread out colors more.
        colors[statename] = cmap(1-np.sqrt((pop-vmin)/(vmax-vmin)))[:3]
    statenames.append(statename)
# cycle through state names, color each one.
ax = plt.gca() # get current axes instance
for nshape,seg in enumerate(m.states):
    # skip DC and Puerto Rico.
    if statenames[nshape] not in ['District of Columbia','Puerto Rico']:
        color = rgb2hex(colors[statenames[nshape]]) 
        poly = Polygon(seg,facecolor=color,edgecolor=color)
        ax.add_patch(poly)
plt.title('Coronavirus Mask Intent May')
plt.show()


# Lambert Conformal map of lower 48 states.
m = Basemap(llcrnrlon=-119,llcrnrlat=22,urcrnrlon=-64,urcrnrlat=49,
        projection='lcc',lat_1=33,lat_2=45,lon_0=-95)
shp_info = m.readshapefile('C:\\Users\\xinli\\Downloads\\st99_d00','states',drawbounds=True)

colors={}
statenames=[]
cmap = plt.cm.hot # use 'hot' colormap
vmin = 0; vmax =450 # set range.
for shapedict in m.states_info:
    statename = shapedict['NAME']
    # skip DC and Puerto Rico.
    if statename not in ['District of Columbia','Puerto Rico']:
        pop = dictionary2[statename]
        # calling colormap with value between 0 and 1 returns
        # rgba value.  Invert color range (hot colors are high
        # population), take sqrt root to spread out colors more.
        colors[statename] = cmap(1-np.sqrt((pop-vmin)/(vmax-vmin)))[:3]
    statenames.append(statename)
# cycle through state names, color each one.
ax = plt.gca() # get current axes instance
for nshape,seg in enumerate(m.states):
    # skip DC and Puerto Rico.
    if statenames[nshape] not in ['District of Columbia','Puerto Rico']:
        color = rgb2hex(colors[statenames[nshape]]) 
        poly = Polygon(seg,facecolor=color,edgecolor=color)
        ax.add_patch(poly)
plt.title('Coronavirus Mask Intent June')
plt.show()


difference  = pd.concat([may4, june4], axis=1, join='inner')
difference['Score_diff'] = june4['coronavirusIntent_Mask'] - may4['coronavirusIntent_Mask'] 
difference = (difference-difference.min())/(difference.max()-difference.min())
dictionary2 = difference['Score_diff'].to_dict()
# Lambert Conformal map of lower 48 states.
m = Basemap(llcrnrlon=-119,llcrnrlat=22,urcrnrlon=-64,urcrnrlat=49,
        projection='lcc',lat_1=33,lat_2=45,lon_0=-95)
shp_info = m.readshapefile('C:\\Users\\xinli\\Downloads\\st99_d00','states',drawbounds=True)

colors={}
statenames=[]
cmap = plt.cm.hot # use 'hot' colormap
vmin = 0; vmax =450 # set range.
norm = Normalize(vmin=vmin, vmax=vmax)
mapper = ScalarMappable(norm=norm, cmap=cmap)
for shapedict in m.states_info:
    statename = shapedict['NAME']
    # skip DC and Puerto Rico.
    if statename not in ['District of Columbia','Puerto Rico']:
        pop = dictionary2[statename]
        colors[statename] = cmap(20*np.sqrt((pop-vmin)/(vmax-vmin)))[:3]
    statenames.append(statename)
# cycle through state names, color each one.
ax = plt.gca() # get current axes instance
for nshape,seg in enumerate(m.states):
    # skip DC and Puerto Rico.
    if statenames[nshape] not in ['District of Columbia','Puerto Rico']:
        color = rgb2hex(colors[statenames[nshape]]) 
        poly = Polygon(seg,facecolor=color,edgecolor=color)
        ax.add_patch(poly)
plt.title('Coronavirus Mask Intent Difference Between May and June')
plt.show()

difference2  = pd.concat([may4, june4], axis=1, join='inner')
difference2['Score_diff'] =  may4['coronavirusIntent_Mask'] - june4['coronavirusIntent_Mask'] 
difference2 = (difference2-difference2.min())/(difference2.max()-difference2.min())
dictionary3 = difference2['Score_diff'].to_dict()
# Lambert Conformal map of lower 48 states.
m = Basemap(llcrnrlon=-119,llcrnrlat=22,urcrnrlon=-64,urcrnrlat=49,
        projection='lcc',lat_1=33,lat_2=45,lon_0=-95)
shp_info = m.readshapefile('C:\\Users\\xinli\\Downloads\\st99_d00','states',drawbounds=True)

colors={}
statenames=[]
cmap = plt.cm.hot # use 'hot' colormap
vmin = 0; vmax =450 # set range.
norm = Normalize(vmin=vmin, vmax=vmax)
mapper = ScalarMappable(norm=norm, cmap=cmap)
for shapedict in m.states_info:
    statename = shapedict['NAME']
    # skip DC and Puerto Rico.
    if statename not in ['District of Columbia','Puerto Rico']:
        pop = dictionary3[statename]
        colors[statename] = cmap(20*np.sqrt((pop-vmin)/(vmax-vmin)))[:3]
    statenames.append(statename)
# cycle through state names, color each one.
ax = plt.gca() # get current axes instance
for nshape,seg in enumerate(m.states):
    # skip DC and Puerto Rico.
    if statenames[nshape] not in ['District of Columbia','Puerto Rico']:
        color = rgb2hex(colors[statenames[nshape]]) 
        poly = Polygon(seg,facecolor=color,edgecolor=color)
        ax.add_patch(poly)
plt.title('Coronavirus Mask Intent Difference Between June and April')
plt.show()

states = [
    'Alabama_UnitedStates','Alaska_UnitedStates','Arizona_UnitedStates',
    'Arkansas_UnitedStates','California_UnitedStates','Colorado_UnitedStates',
    'Connecticut_UnitedStates','Delaware_UnitedStates','DistrictofColumbia_UnitedStates',
    'Florida_UnitedStates','Georgia_UnitedStates','Hawaii_UnitedStates',
    'Idaho_UnitedStates','Illinois_UnitedStates','Indiana_UnitedStates',
    'Iowa_UnitedStates','Kansas_UnitedStates','Kentucky_UnitedStates',
    'Louisiana_UnitedStates','Maine_UnitedStates','Maryland_UnitedStates',
    'Massachusetts_UnitedStates','Michigan_UnitedStates','Minnesota_UnitedStates',
    'Mississippi_UnitedStates','Missouri_UnitedStates','Montana_UnitedStates',
    'Nebraska_UnitedStates','Nevada_UnitedStates','NewHampshire_UnitedStates',
    'NewJersey_UnitedStates','NewMexico_UnitedStates','NewYork_UnitedStates',
    'NorthCarolina_UnitedStates','NorthDakota_UnitedStates','Ohio_UnitedStates',
    'Oklahoma_UnitedStates','Oregon_UnitedStates','Pennsylvania_UnitedStates',
    'PuertoRico_UnitedStates','RhodeIsland_UnitedStates','SouthCarolina_UnitedStates',
    'SouthDakota_UnitedStates','Tennessee_UnitedStates','Texas_UnitedStates',
    'Utah_UnitedStates','Vermont_UnitedStates','Virginia_UnitedStates',
    'Washington_UnitedStates','WestVirginia_UnitedStates','Wisconsin_UnitedStates',
    'Wyoming_UnitedStates'
]

today = '2020-11-11'
metrics = [
    "JHU_ConfirmedCases",
    "JHU_ConfirmedDeaths"
]

complete_timeseries = c3aidatalake.evalmetrics(
    "outbreaklocation",
    {
        "spec" : {
            "ids" : states,
            "expressions" : metrics,
            "start" : "2020-02-15",
            "end" : today,
            "interval" : "DAY",
        }
    },
    get_all = True
)

state_from_location = lambda x: "_".join(x.split('_')[-2:]).replace("_UnitedStates", "")
def reshapeTimeseries(timeseries_df):

    reshaped_ts = pd.melt(
        timeseries_df, 
        id_vars=['dates'], 
        value_vars=[x for x in timeseries_df.columns if re.match('.*\.data', x)]
    ).rename(columns={"value": "data", "dates": "date"})

    reshaped_ts["state"] = (
        reshaped_ts["variable"]
        .str.replace("\..*", "")
        .apply(state_from_location)
    )

    reshaped_ts["metric"] = (
        reshaped_ts["variable"]
        .str.replace(".*UnitedStates\.", "")
        .str.replace("\..*", "")
    )
    

    return reshaped_ts
state_timeseries = reshapeTimeseries(complete_timeseries)
state_timeseries.head()
state_cases = (
    state_timeseries
    .loc[state_timeseries.date > '2020-03-10']
    .groupby(['date', 'state', 'metric'])['data']
    .sum()
    .unstack('metric')
    .reset_index()
)
state_cases['death_rate'] = state_cases.apply(
    lambda x: 0 if x.JHU_ConfirmedCases == 0
    else x.JHU_ConfirmedDeaths / x.JHU_ConfirmedCases,
    axis=1
)

maycount = state_cases.loc[state_cases['date']<'2020-05-16']
junecount = state_cases.loc[state_cases['date']>='2020-05-16']
junecount = state_cases.loc[state_cases['date']<'2020-06-12']
alabamajune= junecount[junecount['state']=='Alabama']

#plotting a specific state
from matplotlib.dates import DateFormatter
# Define the date format
date_form = DateFormatter("%m-%d")
ax.xaxis.set_major_formatter(date_form)
plt.show()

nhjune= junecount[junecount['state']=='NewHampshire']
# Create figure and plot space
fig, ax = plt.subplots(figsize=(12, 12))
# Add x-axis and y-axis
ax.scatter(nhjune['date'],
       nhjune['JHU_ConfirmedCases'],
       color='purple')

# Set title and labels for axes
ax.set(xlabel="Date",
       ylabel="Case count",
       title="New Hampshire case count over time")

#plotting all states 
grouped = state_cases.groupby('state')
ncols=4
nrows = int(np.ceil(grouped.ngroups/ncols))

fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=(20,20), sharey=True, squeeze = False)

for (key, ax) in zip(grouped.groups.keys(), axes.flatten()):
    grouped.get_group(key).plot(ax=ax, x='date',y='JHU_ConfirmedCases', label = key)
plt.title("Case data")
plt.show()

state_cases = state_cases.loc[state_cases['date']<'2020-06-12']

#plotting all states(before may)
grouped = state_cases.groupby('state')
ncols=4
nrows = int(np.ceil(grouped.ngroups/ncols))

fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=(20,20), sharey=True, squeeze = False)

for (key, ax) in zip(grouped.groups.keys(), axes.flatten()):
    grouped.get_group(key).plot(ax=ax, x='date',y='JHU_ConfirmedCases', label = key)
plt.title("Case data")
plt.show()

state_cases=state_cases.loc[state_cases['date']>='2020-05-16']

#obtaining census data from 2018
population_limits = (
    f"contains(parent, 'UnitedStates') &&" # US data
    "gender == 'Male/Female' && year == 2018 && origin == 'United States Census'" # From 2018 estimates
)

census = c3aidatalake.fetch(
    "populationdata",
    {
      "spec": {
        "filter": population_limits
      }
    },
    get_all = True
)

census['state'] = census['parent.id'].apply(state_from_location)
census = census.rename(columns={'parent.id': 'location'})
census_cols = [
    "populationAge",
    "value",
    "location",
    "state"
]

census_by_state = (
    census[census_cols]
    .loc[census.state.isin(map(lambda x: x.replace("_UnitedStates", ""), states))]
    .groupby(["state", "populationAge"])['value']
    .sum()
    .unstack("populationAge")
    .reset_index()
)

merged_cases = pd.merge(left=census_by_state, right=state_cases, left_on='state', right_on='state')
merged_cases['cases_per_pop']=merged_cases['JHU_ConfirmedCases']/merged_cases['Total']

grouped = merged_cases.groupby('state')
ncols=4
nrows = int(np.ceil(grouped.ngroups/ncols))

fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=(20,20), sharey=True, squeeze = False)

for (key, ax) in zip(grouped.groups.keys(), axes.flatten()):
    grouped.get_group(key).plot(ax=ax, x='date',y='cases_per_pop', label = key)
plt.title("Case data")
plt.show()

rslt_df2 = survey.loc[survey['startTime'] < '2020-06-12'] 
rslt_df2['startTime'] = pd.to_datetime(rslt_df2['startTime'])
rslt_df2['startTime'] = rslt_df2['startTime'].dt.strftime('%Y-%m-%d %H:%M:%S')
rslt_df2['startTime'] = pd.to_datetime(rslt_df2['startTime'])
rslt_df2["location.id"] = rslt_df2["location.id"].str.replace("_UnitedStates", "")
merged_cases= pd.merge(left=merged_cases,right=rslt_df2, left_on=['state','date'],right_on=['location.id','startTime'])

average = (merged_cases.groupby([ 'state'], as_index=False)['coronavirusConcern','coronavirusIntent_SixFeet','coronavirusIntent_Mask','coronavirusIntent_WashHands'].mean())
average = pd.merge(average,merged_cases[['state','cases_per_pop']],on='state', how='left')
average =average.drop_duplicates()

average1=pd.DataFrame(average.groupby('state')['coronavirusIntent_Mask'].mean()).reset_index()
average2=pd.DataFrame(average.groupby('state')['cases_per_pop'].mean())
x = average2['cases_per_pop'].values
y = average1['coronavirusIntent_Mask'].values
types = average1['state'].values
fig, ax = plt.subplots(figsize=(10,10))
ax.scatter(x, y)

ax.set_ylabel('covid Concern', fontsize=14)
ax.set_xlabel('cases_per_pop', fontsize=14)
ax.set_title('Covid Mask Intent vs Cases per population', fontsize=18)

for i, txt in enumerate(types):
    ax.annotate(txt, (x[i], y[i]), xytext=(10,10), textcoords='offset points')
    plt.scatter(x, y, marker='x', color='red')
 
average_may = (merged_cases.loc[merged_cases['date'] < '2020-05-12'] .groupby([ 'state'], as_index=False)['coronavirusConcern','coronavirusIntent_SixFeet','coronavirusIntent_Mask','coronavirusIntent_WashHands'].mean())
average_after = (merged_cases.loc[merged_cases['date'] >= '2020-05-12'] .groupby([ 'state'], as_index=False)['coronavirusConcern','coronavirusIntent_SixFeet','coronavirusIntent_Mask','coronavirusIntent_WashHands'].mean())
average_may = pd.merge(average_may,merged_cases[['state','cases_per_pop']],on='state', how='left')
average =average.drop_duplicates()
#obtaining mobility data 
mobility_trends = c3aidatalake.evalmetrics(
    "outbreaklocation",
    {
        "spec" : {
            "ids" : ["California_UnitedStates", "Texas_UnitedStates"],
            "expressions" : [
                "Apple_WalkingMobility", 
                "Apple_DrivingMobility",
                "Google_ParksMobility",
                "Google_ResidentialMobility"
              ],
            "start" : "2020-05-16",
            "end" : "2020-06-12",
            "interval" : "DAY",
        }
    },
    get_all = True
)

policy_united_states = c3aidatalake.fetch(
  "locationpolicysummary",
  {
      "spec" : {
          "filter" : "contains(location.id, 'UnitedStates')",
          "limit" : -1
      }
  }
)

versions = c3aidatalake.read_data_json(
    "locationpolicysummary",
    "allversionsforpolicy",
    {
        "this" : {
            "id" : "Wisconsin_UnitedStates_Policy"
        }
    }
)

pd.json_normalize(versions)

dataframe=pd.read_csv("C:\\Users\\xinli\\Downloads\\corona_tweets_51.csv", header=None)
dataframe=dataframe[0]
dataframe.to_csv("ready_corona_tweets_51.csv", index=False, header=None)

#replace empty locations with nan and remove them
dataframe=pd.read_csv("C:\\Users\\xinli\\Downloads\\ready_corona_tweets_59.csv", header=None)
dataframe2=pd.read_csv("C:\\Users\\xinli\\Downloads\\corona_tweets_59.csv", header=None)
dataframe2.columns = ['id','sentiment']
dataframe2['id']=dataframe2['id'].astype(np.int64)
dataframe.columns = dataframe.iloc[0]
dataframe = dataframe[1:]
dataframe['user_location'].replace('', np.nan, inplace=True)
dataframe.dropna(subset=['user_location'], inplace=True)
dataframe['id']=dataframe['id'].astype(np.int64)

merged_inner = pd.merge(left=dataframe, right=dataframe2, left_on='id', right_on='id')
count_tweets= merged_inner['user_location'].value_counts()
# could replace this with a dictionary, I also don't think it's 100% accurate since the abbreviations could be used
#for other things
merged_inner.loc[merged_inner['user_location'].str.contains('California|CA'), 'user_location'] = 'California'
merged_inner.loc[merged_inner['user_location'].str.contains('Texas|TX'), 'user_location'] = 'Texas'
merged_inner.loc[merged_inner['user_location'].str.contains('NY|New York'), 'user_location'] = 'New York'
merged_inner.loc[merged_inner['user_location'].str.contains('Texas|TX'), 'user_location'] = 'Texas'
merged_inner.loc[merged_inner['user_location'].str.contains('Minnesota'), 'user_location'] = 'Minnesota'
merged_inner.loc[merged_inner['user_location'].str.contains('Oregon'), 'user_location'] = 'Oregon'
merged_inner.loc[merged_inner['user_location'].str.contains('AZ| Arizona'), 'user_location'] = 'Arizona'
merged_inner.loc[merged_inner['user_location'].str.contains('AK|Alaska'), 'user_location'] = 'Alaska'
merged_inner.loc[merged_inner['user_location'].str.contains('CO|Colorado'), 'user_location'] = 'Colorado'
merged_inner.loc[merged_inner['user_location'].str.contains('Delaware|DE'), 'user_location'] = 'Delaware'
merged_inner.loc[merged_inner['user_location'].str.contains('Florida|FL'), 'user_location'] = 'Florida'
merged_inner.loc[merged_inner['user_location'].str.contains('Georgia|GA'), 'user_location'] = 'Georgia'


merged_inner = merged_inner[pd.to_numeric(merged_inner['retweet_count'])>5]
merged_inner = merged_inner.drop(columns=['user_default_profile_image', 'in_reply_to_user_id', 'in_reply_to_status_id','in_reply_to_screen_name', 'user_time_zone','coordinates','media'])

#merged_inner=merged_inner[merged_inner.groupby('user_location').user_location.transform('count') > 1]
