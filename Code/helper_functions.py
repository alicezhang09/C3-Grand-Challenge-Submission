# -*- coding: utf-8 -*-
"""
Created on Tue Nov 17 18:47:49 2020

@author: Alice Zhang
"""

import os 
import json
import requests
import pandas as pd
import re
import numpy as np

#helper functions and setting up twitter API
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

def reshapeTimeseries(timeseries_df):
    state_from_location = lambda x: "_".join(x.split('_')[-2:]).replace("_UnitedStates", "")
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

#function to convert categorical data (e.g. race, gender) into dichotomous variables
def value_to_col(dataframe, policy):
    temp = dataframe[policy].fillna(0)
    dfcount = temp.value_counts()
    options = list(dfcount.to_frame().index.values)
    for i in options:
        dataframe.loc[:,policy+i]=np.where(temp.str.contains(i),1,0)
    return dataframe

#function to format states to match state names to plot using basemap
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

# convert user locations from tweet data into US states
# could replace this with a dictionary, I also don't think it's 100% accurate since the abbreviations could be used
# for other things
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