# -*- coding: utf-8 -*-
"""
Created on Mon Apr 17 14:00:59 2017

@author: brendanbailey
"""

import pandas as pd

def eda(dataframe): #Performs exploratory data analysis on the dataframe
    print "columns \n", dataframe.columns
    print "head \n", dataframe.head()
    print "tail \n", dataframe.tail()
    print "missing values \n", dataframe.isnull().sum()
    print "dataframe types \n", dataframe.dtypes
    print "dataframe shape \n", dataframe.shape
    print "dataframe describe \n", dataframe.describe() #summary statistics
    for item in dataframe:
        print item
        print dataframe[item].nunique()
    print "%s duplicates out of %s records" % (len(dataframe) - len(dataframe.drop_duplicates()), len(dataframe))

def clean_and_transform(train, spray, weather):
    #Imputing 7:44:32 PM because that time has the most frequency and they all occur on 2011-09-07. Same date as when the null values are.
    spray.Time.fillna("7:44:32 PM", inplace = True)
    
    #Rolling up Data so only one record for date and trap available
    train["Date-Trap"] = train["Date"] + "-" + train["Trap"]
    target_remap_dict = {}
    for index, row in train[["Date-Trap", "WnvPresent", "Species"]].iterrows():
        if row["Species"] in ["CULEX PIPIENS/RESTUANS", "CULEX RESTUANS", "CULEX PIPIENS"]:
            wnv_mosquitos = 1
        else:
            wnv_mosquitos = 0
        try:
            if target_remap_dict[row["Date-Trap"]]["WnvPresentAdj"] == 0:
                target_remap_dict[row["Date-Trap"]]["WnvPresentAdj"] = row["WnvPresent"]
            if target_remap_dict[row["Date-Trap"]]["WnvMosquito"] == 0:
                target_remap_dict[row["Date-Trap"]]["WnvMosquito"] = wnv_mosquitos
        except KeyError: 
            target_remap_dict[row["Date-Trap"]] = {"WnvPresentAdj": row["WnvPresent"], "WnvMosquito": wnv_mosquitos}
    target_remap_df = pd.DataFrame.from_dict(target_remap_dict, orient = "index")
    
    #Combining DFs
    master_df = train.drop_duplicates(subset = ["Date", "Trap"])
    del master_df['WnvPresent']
    del master_df['Species']
    master_df = pd.merge(master_df, target_remap_df, left_on = "Date-Trap", right_index = True)
    master_df = pd.merge(master_df, weather[weather.Station == 2], left_on = "Date", right_on = "Date", how = "left")
    del master_df["Date-Trap"]
    
    return master_df

train_df = pd.read_csv("assets/train.csv")
weather_df = pd.read_csv("assets/weather.csv")
spray_df = pd.read_csv("assets/spray.csv")
master_df = clean_and_transform(train_df, spray_df, weather_df)