# -*- coding: utf-8 -*-
"""
Created on Fri Nov  6 19:18:43 2020

@author: xinli
"""

#import packages
import os 
import json
import requests
import pandas as pd
from matplotlib import pyplot as plt
import numpy as np
import re
from matplotlib.cm import ScalarMappable
from mpl_toolkits.basemap import Basemap as Basemap
from matplotlib.colors import rgb2hex,Normalize
from matplotlib.patches import Polygon

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

# documentation can be found in helper_functions file
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

def abb_to_state(dataframe,abb, state):
    dataframe.loc[dataframe['user_location'].str.contains(abb),'user_location']=state
def filter_tweets(dataframe,dataframe2):
    dataframe2.columns = ['id','sentiment']
    dataframe2['id']=dataframe2['id'].astype(np.int64)
    dataframe.columns = dataframe.iloc[0]
    dataframe = dataframe[1:]
    dataframe['user_location'].replace('', np.nan, inplace=True)
    dataframe.dropna(subset=['user_location'], inplace=True)
    dataframe['id']=dataframe['id'].astype(np.int64)
    states_abb = ["AL", "AK", "AZ", "AR", "CA", "CO", "CT", "DC", "DE", "FL", "GA", 
              "HI", "ID", "IL", "IN", "IA", "KS", "KY", "ME", "MD", 
              "MA", "MI", "MN", "MS", "MO", "MT", "NE", "NV", "NH", "NJ", 
              "NM", "NY", "NC", "ND", "OH", "OK", "OR", "PA", "RI", "SC", 
              "SD", "TN", "TX", "UT", "VT", "VA", "WA", "WV", "WI", "WY"]
    state_names = ["Alaska", "Alabama", "Arkansas", "Arizona", "California", "Colorado", 
                   "Connecticut", "District of Columbia", "Delaware", "Florida", "Georgia",  
                   "Hawaii", "Iowa", "Idaho", "Illinois", "Indiana", "Kansas", "Kentucky", "Louisiana", 
                   "Massachusetts", "Maryland", "Maine", "Michigan", "Minnesota", "Missouri", "Mississippi", 
                   "Montana", "North Carolina", "North Dakota", "Nebraska", "New Hampshire", "New Jersey", 
                   "New Mexico", "Nevada", "New York", "Ohio", "Oklahoma", "Oregon", "Pennsylvania", "Puerto Rico", 
                   "Rhode Island", "South Carolina", "South Dakota", "Tennessee", "Texas", "Utah", "Virginia",  
                   "Vermont", "Washington", "Wisconsin", "West Virginia", "Wyoming"]
    states2= state_names + states_abb  
    dataframe = dataframe[dataframe['user_location'].isin(states2)]
    dataframe3= dataframe[['created_at','id','retweet_count','user_followers_count','user_location','user_verified']]
    merged_inner = pd.merge(left=dataframe3, right=dataframe2, left_on='id', right_on='id')
    abb_to_state(merged_inner, 'CA|LA','California')
    abb_to_state(merged_inner, 'TX','Texas')
    abb_to_state(merged_inner, 'NY|New York','NewYork')
    abb_to_state(merged_inner, 'AL','Alaska')
    abb_to_state(merged_inner, 'AK','Arkansas')
    abb_to_state(merged_inner, 'AZ','Arizona')
    abb_to_state(merged_inner, 'CO','Colorado')
    abb_to_state(merged_inner, 'CT','Connecticut')
    abb_to_state(merged_inner, 'DC|District of Columbia','DistrictOfColumbia')
    abb_to_state(merged_inner, 'DE','Delaware')
    abb_to_state(merged_inner, 'FL','Florida')
    abb_to_state(merged_inner, 'GA','Georgia')
    abb_to_state(merged_inner, 'HI','Hawaii')
    abb_to_state(merged_inner, 'ID','Idaho')
    abb_to_state(merged_inner, 'IN','Indiana')
    abb_to_state(merged_inner, 'IL','Illinois')
    abb_to_state(merged_inner, 'IA','Iowa')
    abb_to_state(merged_inner, 'KS','Kansas')
    abb_to_state(merged_inner, 'KY','Kentucky')
    abb_to_state(merged_inner, 'ME','Maine')
    abb_to_state(merged_inner, 'MD','Maryland')
    abb_to_state(merged_inner, 'MI','Michigan')
    abb_to_state(merged_inner, 'MN','Minnesota')
    abb_to_state(merged_inner, 'MS','Mississippi')
    abb_to_state(merged_inner, 'MO','Missouri')
    abb_to_state(merged_inner, 'MT','Montana')
    abb_to_state(merged_inner, 'NE','Nebraska')
    abb_to_state(merged_inner, 'NH|New Hampshire','NewHampshire')
    abb_to_state(merged_inner, 'NJ|New Jersey','NewJersey')
    abb_to_state(merged_inner, 'NV','Nevada')
    abb_to_state(merged_inner, 'NM|New Mexico','NewMexico')
    abb_to_state(merged_inner, 'ND|North Dakota','NorthDakota')
    abb_to_state(merged_inner, 'NC|North Carolina','NorthCarolina')
    abb_to_state(merged_inner, 'OH','Ohio')
    abb_to_state(merged_inner, 'OK','Oklahoma')
    abb_to_state(merged_inner, 'OR','Oregon')
    abb_to_state(merged_inner, 'PA','Pennsylvania')
    abb_to_state(merged_inner, 'RI|Rhode Island','Rhode Island')
    abb_to_state(merged_inner, 'SC|South Carolina','SouthCarolina')
    abb_to_state(merged_inner, 'SD|South Dakota','SouthDakota')
    abb_to_state(merged_inner, 'TN','Tennessee')
    abb_to_state(merged_inner, 'UT','Utah')
    abb_to_state(merged_inner, 'VT','Vermont')
    abb_to_state(merged_inner, 'VA','Virginia')
    abb_to_state(merged_inner, 'WV|West Virginia','West Virginia')
    abb_to_state(merged_inner, 'WI','Wisconsin')
    abb_to_state(merged_inner, 'WY','Wyoming')
    abb_to_state(merged_inner, 'WA','Washington')
    return merged_inner

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

def value_to_col(dataframe, policy):
    temp = dataframe[policy].fillna(0)
    dfcount = temp.value_counts()
    options = list(dfcount.to_frame().index.values)
    for i in options:
        dataframe.loc[:,policy+i]=np.where(temp.str.contains(i),1,0)
    return dataframe

def format_states(dataframe, colname):
    dataframe = dataframe[dataframe[colname].notna()]
    dataframe[colname] = dataframe[colname].str.replace("_UnitedStates", "")
    dataframe[colname] = dataframe[colname].str.replace("SouthCarolina", "South Carolina")
    dataframe[colname] = dataframe[colname].str.replace("NorthCarolina", "North Carolina")
    dataframe[colname] = dataframe[colname].str.replace("DistrictofColumbia", "District of Columbia")
    dataframe[colname] = dataframe[colname].str.replace("NewHampshire", "New Hampshire")
    dataframe[colname] = dataframe[colname].str.replace("SouthDakota", "South Dakota")
    dataframe[colname] = dataframe[colname].str.replace("NorthDakota", "North Dakota")
    dataframe[colname] = dataframe[colname].str.replace("NewYork", "New York")
    dataframe[colname] = dataframe[colname].str.replace("NewMexico", "New Mexico")
    dataframe[colname] = dataframe[colname].str.replace("PuertoRico", "Puerto Rico")
    dataframe[colname] = dataframe[colname].str.replace("WestVirginia", "West Virginia")
    dataframe[colname] = dataframe[colname].str.replace("NewJersey", "New Jersey")
    dataframe[colname] = dataframe[colname].str.replace("RhodeIsland", "Rhode Island")
    return dataframe

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

# Iterate and save tweets with the phrase "flatten the curve"
# Twitter public API gives you access to the past 7 days of tweets
i=0
df = pd.DataFrame(columns=['Name', 'Location', 'Time'])
for tweet in tweets:
    df.loc[i]= [tweet.user.screen_name, tweet.user.location, tweet.created_at]
    i=i+1

#replace empty locations with nan and remove them
df['Location'].replace('', np.nan, inplace=True)
df.dropna(subset=['Location'], inplace=True)

# get occurences of user locations
count = df['Location'].value_counts()

#get all available survey data
survey = c3aidatalake.fetch(
    "surveydata",
    {
        "spec": {
            
        }
    },
    get_all = True
)

#format time column in survey data
survey['startTime'] = pd.to_datetime(survey['startTime'])
survey['startTime'] = survey['startTime'].apply(lambda t: t.replace(second=0))
survey['startTime'] = survey['startTime'].apply(lambda t: t.replace(minute=0))
survey['startTime'] = survey['startTime'].apply(lambda t: t.replace(hour=0))

#making subset of survey data from after june and before may 16
rslt_df = survey.loc[survey['startTime'] > '2020-06-06'] 
rslt_df2 = survey.loc[survey['startTime'] < '2020-05-16'] 

#reformatting location
rslt_df2["location.id"] = rslt_df2["location.id"].str.replace("_UnitedStates", "")

#s.environ["PROJ_LIB"] = "C:\\Users\\xinli\\Anaconda3\\Library\\share"; #fixr

#subset with only location and mask intent before May 16
may2 = rslt_df2[["location.id","coronavirusIntent_Mask"]]
count = may2['location.id'].value_counts()

may2=format_states(may2,"location.id")
#compute average
may_average=pd.DataFrame(may2.groupby('location.id')['coronavirusIntent_Mask'].mean())
dictionary_may_varage = may_average['coronavirusIntent_Mask'].to_dict()

#subset with only location and mask intent after june 5
june2 = rslt_df[["location.id","coronavirusIntent_Mask"]]
count = june2['location.id'].value_counts()
june2 = format_states(june2, "location.id")

june4=pd.DataFrame(june2.groupby('location.id')['coronavirusIntent_Mask'].mean())
junesurvey=pd.DataFrame(june2.groupby('location.id',as_index=False)['coronavirusIntent_Mask'].mean())
dictionary_june_average = june4['coronavirusIntent_Mask'].to_dict()

#plotting difference between average survey response to coronavirusIntent_Mask between June and May
difference2  = pd.concat([may_average, june4], axis=1, join='inner')
difference2['Score_diff'] =  may_average['coronavirusIntent_Mask'] - june4['coronavirusIntent_Mask'] 
difference2 = (difference2-difference2.min())/(difference2.max()-difference2.min())
dictionary_difference = difference2['Score_diff'].to_dict()

# Lambert Conformal map of lower 48 states.
m = Basemap(llcrnrlon=-119,llcrnrlat=22,urcrnrlon=-64,urcrnrlat=49,
        projection='lcc',lat_1=33,lat_2=45,lon_0=-95)
shp_info = m.readshapefile('C:\\Users\\xinli\\Downloads\\st99_d00','states',drawbounds=True)

colors={}
statenames=[]
cmap = plt.cm.hot # use 'hot' colormap
vmin = 0; vmax = 450 # set range.
norm = Normalize(vmin=vmin, vmax=vmax)
mapper = ScalarMappable(norm=norm, cmap=cmap)
for shapedict in m.states_info:
    statename = shapedict['NAME']
    # skip DC and Puerto Rico.
    if statename not in ['District of Columbia','Puerto Rico']:
        pop = dictionary_difference[statename]
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

#getting case count data
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

#looking at more specific ranges
maycount = state_cases.loc[state_cases['date']<'2020-05-16']
junecount = state_cases.loc[state_cases['date']>='2020-05-16']
junecount = state_cases.loc[state_cases['date']<'2020-06-12']

#plotting a specific state
from matplotlib.dates import DateFormatter
# Define the date format
date_form = DateFormatter("%m-%d")
ax.xaxis.set_major_formatter(date_form)
plt.show()

#plotting all state case count over time
grouped = state_cases.groupby('state')
ncols=4
nrows = int(np.ceil(grouped.ngroups/ncols))
fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=(20,20), sharey=True, squeeze = False)

for (key, ax) in zip(grouped.groups.keys(), axes.flatten()):
    grouped.get_group(key).plot(ax=ax, x='date',y='JHU_ConfirmedCases', label = key)
plt.title("Case data")
plt.show()

#getting state cases befoere june 12
state_cases_early = state_cases.loc[state_cases['date']<'2020-06-12']

#plotting all states(before may)
grouped = state_cases_early.groupby('state')
ncols=4
nrows = int(np.ceil(grouped.ngroups/ncols))

fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=(20,20), sharey=True, squeeze = False)

for (key, ax) in zip(grouped.groups.keys(), axes.flatten()):
    grouped.get_group(key).plot(ax=ax, x='date',y='JHU_ConfirmedCases', label = key)
plt.title("Case data")
plt.show()

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

#reformatting census data
import datetime
census_by_state2=census_by_state
census_and_cases = pd.merge(left=census_by_state2, right = state_cases, left_on='state',right_on='state')
census_and_cases['state'] =  census_and_cases['state'].astype(str) + '_UnitedStates'

# adding column to match survey dates with policies in place in that particular state
surveydates= survey['startTime'].value_counts()
case_dates = list(census_and_cases['date'].value_counts().to_frame().index.values)
main_list = list(set(case_dates)-set(surveydates.to_frame().index.values))
for i in main_list:
  census_and_cases = census_and_cases[census_and_cases['date'] != i]
  
census_num = census_and_cases.select_dtypes(include=[np.number])
normalized_census = ((census_num-census_num.min())/(census_num.max()-census_num.min()))
census_and_cases[normalized_census.columns] = normalized_census

# plotting cases per population of all states before mid June
merged_cases = pd.merge(left=census_by_state, right=state_cases, left_on='state', right_on='state')
merged_cases['cases_per_pop']=merged_cases['JHU_ConfirmedCases']/merged_cases['Total']
merged_cases['cases_per_pop'] = merged_cases['cases_per_pop']/max(merged_cases['cases_per_pop'])

grouped = merged_cases.groupby('state')
ncols=4
nrows = int(np.ceil(grouped.ngroups/ncols))

fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=(20,20), sharey=True, squeeze = False)

for (key, ax) in zip(grouped.groups.keys(), axes.flatten()):
    grouped.get_group(key).plot(ax=ax, x='date',y='cases_per_pop', label = key)
plt.title("Case data")
plt.show()

#subsetting survey data to before early June
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
 

df = merged_cases[['date','state','coronavirusIntent_SixFeet']]
df=df.groupby(['date','state'],as_index=False).mean()
df['coronavirusIntent_SixFeet']=df['coronavirusIntent_SixFeet']/100

merged_cases = pd.merge(left=census_by_state, right=state_cases, left_on='state', right_on='state')
merged_cases['cases_per_pop']=merged_cases['JHU_ConfirmedCases']/merged_cases['Total']
merged_cases = merged_cases.loc[merged_cases['date'] > '2020-04-30'] 
merged_cases = merged_cases.loc[merged_cases['date'] < '2020-06-08'] 
merged_cases['cases_per_pop'] = (merged_cases['cases_per_pop']-min(merged_cases['cases_per_pop']))/(max(merged_cases['cases_per_pop'])-min(merged_cases['cases_per_pop']))

grouped = merged_cases.groupby('state')
grouped2 = df.groupby('state')
ncols=7
nrows = int(np.ceil(grouped2.ngroups/ncols))

fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=(20,20), sharey=True, squeeze = False)
plt.subplots_adjust(left=0.125, bottom=0.4, right=0.9, top=1.2, wspace=0.3, hspace=1.2)


for (key, ax) in zip(grouped2.groups.keys(), axes.flatten()):
    grouped2.get_group(key).plot(ax=ax, x='date',y='coronavirusIntent_SixFeet',title =key)
    grouped.get_group(key).plot(ax=ax, x='date',y='cases_per_pop')
    lgd = ax.legend(loc='upper center', bbox_to_anchor=(0.5,-0.35))
    date_form = DateFormatter("%m-%d")
    ax.xaxis.set_major_formatter(date_form)

plt.title("Case data")
plt.show()


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
          "include": "stayAtHome, mandatoryQuarantine, largeGatherings,schoolClosure,easingOrder, emergencyDeclaration",
          "limit" : -1
      }
  }
)

policy_united_states = c3aidatalake.fetch(
  "locationpolicysummary",
  {
      "spec" : {
          "filter" : "contains(location.id, 'UnitedStates')",
          "include": "stayAtHome, mandatoryQuarantine, largeGatherings,schoolClosure,easingOrder, emergencyDeclaration",
          "limit" : -1
      }
  }
)


states_list = policy_united_states['id'].tolist()
states_list.pop(0)
history_policies = []
for state in states_list:
  print(state)
  if state == 'United States_UnitedStates_Policy':
    continue
  try:
    policy_state = c3aidatalake.read_data_json(
        "locationpolicysummary",
        "allversionsforpolicy",
        {
        "this": {
            "id": state
        }
    }
    )
  except:
    print("none")

  history_policies = history_policies + [policy_state]
  
df = pd.DataFrame(list(history_policies[0][2].items()),columns = ['column1','column2']).T
df = df.rename(columns=df.iloc[0])
df = df.drop(df.index[0])

df3= df[['location', 'lastSavedTimestamp','stayAtHome', 'mandatoryQuarantine','largeGatherings','easingOrder','emergencyDeclaration']]
for i in range(len(history_policies)):
    for j in range(len(history_policies[i])):
        df2= pd.DataFrame(list(history_policies[i][j].items()),columns = ['column1','column2']).T
        df2 = df2.rename(columns=df2.iloc[0])
        df2 = df2.drop(df2.index[0])
        df2=df2[['location', 'lastSavedTimestamp','stayAtHome', 'mandatoryQuarantine','largeGatherings','easingOrder','emergencyDeclaration']]
        df3 = df3.append(df2)

df3['stayAtHome'].value_counts()

df6 = value_to_col(df3, "stayAtHome")
df6 = value_to_col(df3, "mandatoryQuarantine")
df6 = value_to_col(df3, "easingOrder")
df6 = value_to_col(df3, "emergencyDeclaration")
df6 = value_to_col(df3, "largeGatherings")

df3 = df3.reset_index()
a=[]
for i in range(len(df3)):
    a.append(df3.loc[i,'location'].get('id'))
    
df3['location.id'] = a
    
dataframe=pd.read_csv("C:\\Users\\xinli\\Downloads\\corona_tweets_51.csv", header=None)
dataframe=dataframe[0]
dataframe.to_csv("ready_corona_tweets_51.csv", index=False, header=None)


#replace empty locations with nan and remove them
dataframe=pd.read_csv("C:\\Users\\xinli\\Downloads\\ready_corona_tweets_51.csv", header=None,error_bad_lines=False, index_col=False, dtype='unicode')
dataframe2=pd.read_csv("C:\\Users\\xinli\\Downloads\\corona_tweets_51.csv", header=None,error_bad_lines=False, index_col=False, dtype='unicode')
dataframe2.columns = ['id','sentiment']
dataframe2['id']=dataframe2['id'].astype(np.int64)
dataframe.columns = dataframe.iloc[0]
dataframe = dataframe[1:]
dataframe['user_location'].replace('', np.nan, inplace=True)
dataframe.dropna(subset=['user_location'], inplace=True)
dataframe['id']=dataframe['id'].astype(np.int64)


List = open("C:\\Users\\xinli\\Downloads\\city_names.txt",).readlines()
del List[0]
del List[1]

jun1=pd.read_csv("C:\\Users\\xinli\\Downloads\\c3-75_june1.csv", header=None,error_bad_lines=False, index_col=False, dtype='unicode')
jun1_sentiment=pd.read_csv("C:\\Users\\xinli\\Downloads\\corona_tweets_75.csv", header=None,error_bad_lines=False, index_col=False, dtype='unicode')
check = filter_tweets(jun1,jun1_sentiment)
jun9=pd.read_csv("C:\\Users\\xinli\\Downloads\\Twitter_Data_June9.csv", header=None,error_bad_lines=False, index_col=False, dtype='unicode')
jun9_sentiment=pd.read_csv("C:\\Users\\xinli\\Downloads\\corona_tweets_83.csv", header=None,error_bad_lines=False, index_col=False, dtype='unicode')
may8=pd.read_csv("C:\\Users\\xinli\\Downloads\\ready_corona_tweets_51.csv", header=None,error_bad_lines=False, index_col=False, dtype='unicode')
may8_sentiment=pd.read_csv("C:\\Users\\xinli\\Downloads\\corona_tweets_51.csv", header=None,error_bad_lines=False, index_col=False, dtype='unicode')
checkmay8=filter_tweets(may8,may8_sentiment)
check = filter_tweets(jun1,jun1_sentiment)
check2 = filter_tweets(jun9,jun9_sentiment)
check = check.append(check2)
check=check.append(checkmay8)
check.to_csv("filtered_tweets.csv", index=False)

count_tweets= check['user_location'].value_counts()
check["created_at"] = check["created_at"].astype('datetime64[ns]') 
check["created_at"] = check.created_at.dt.to_pydatetime()
check["created_at"] = check["created_at"].apply(lambda t: t.replace(second=0))
check["created_at"] = check["created_at"].apply(lambda t: t.replace(minute=0))
check["created_at"] = check["created_at"].apply(lambda t: t.replace(hour=0))
census_and_cases = pd.merge(left=census_by_state2, right = state_cases, left_on='state',right_on='state')
census_and_cases['date'] = census_and_cases['date'].dt.strftime('%Y-%m-%d')
count_tweet_dates = check['created_at'].value_counts()
tweets_and_cases= pd.merge(right=census_and_cases, left = check, right_on=['state','date'],left_on=['user_location','created_at'],sort=False, how = 'left')

tweets_num = tweets_and_cases.select_dtypes(include=[np.number])
tweets_norm= (tweets_num-tweets_num.min())/(tweets_num.max()-tweets_num.min())
tweets_and_cases[tweets_norm.columns] = tweets_norm
tweets_and_cases = tweets_and_cases.drop(columns=['user_location', 'user_verified', 'id'])

tweets_and_cases['']=(merged_cases['cases_per_pop']-min(merged_cases['cases_per_pop']))/(max(merged_cases['cases_per_pop'])-min(merged_cases['cases_per_pop']))

temp2 = []
time2 = list(pd.to_datetime(tweets_and_cases['date']))
for i in range(len(tweets_and_cases)):
  if (time2[i].date()<datetime.date(2020,5,29)):
    temp2.append(datetime.datetime(2020,5,5))
  else:
    temp2.append(datetime.datetime(2020,5,29))

tweet_copy = pd.DataFrame(index=tweets_and_cases.index)
tweet_copy['temp']=temp2
tweets_and_cases['time']=tweet_copy['temp']
tweets_and_cases['Time_Diff'] = (pd.to_datetime(tweets_and_cases.date) - pd.to_datetime(tweets_and_cases.time)).dt.days
tweets_and_cases=tweets_and_cases.dropna()
tweets_and_cases['state']=tweets_and_cases['state']+"_UnitedStates"
states4 = list(df3['location.id'].value_counts().to_frame().index.values)
states5= list(tweets_and_cases['state'].value_counts().to_frame().index.values)
main_list = list(set(states5) - set(states4))
for i in main_list:
  tweets_and_cases = tweets_and_cases[tweets_and_cases['state'] != i]

tweets_and_cases = tweets_and_cases.rename(columns={'state': 'location.id'})  
merged =  tweets_and_cases.merge(df3, on=['time','location.id'], how='left',sort=False)

df3=df3.drop_duplicates(subset=['state','time']) 































































