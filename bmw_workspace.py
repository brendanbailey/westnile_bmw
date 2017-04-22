# -*- coding: utf-8 -*-
"""
Created on Mon Apr 17 14:00:59 2017

@author: brendanbailey
"""

import pandas as pd
import numpy as np
from geopy.distance import vincenty
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

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
    #Replacing M which is missing data and T which is Trace Data with NAN
    weather_df.replace("M", np.nan, inplace = True)
    weather_df.replace("T", np.nan, inplace = True)
    
    #Casting all appropriate weather columns as ints
    numeric_columns = ['Station', 'Tmax', 'Tmin', 'Tavg', 'Depart', 'DewPoint', 'WetBulb', 'Heat', 'Cool', 'Sunrise', 'Sunset', 'Depth', 'Water1', 'SnowFall', 'PrecipTotal', 'StnPressure', 'SeaLevel', 'ResultSpeed', 'ResultDir', 'AvgSpeed']
    for column in numeric_columns:
        weather_df[column] = weather_df[column].apply(pd.to_numeric, errors='coerce')
    
    #Imputing 7:44:32 PM because that time has the most frequency and they all occur on 2011-09-07. Same date as when the null values are.
    spray.Time.fillna("7:44:32 PM", inplace = True)
    
    #Imputing Data for WetBulb and StnPressure
    wetbulb_median = np.median(weather_df.dropna(subset = ["WetBulb"]).WetBulb)
    stnpressure_median = np.median(weather_df.dropna(subset = ["StnPressure"]).StnPressure)
    weather_df.WetBulb.fillna(np.median(wetbulb_median), inplace = True)
    weather_df.StnPressure.fillna(np.median(stnpressure_median), inplace = True)
    
    #Originally I tried using dataframe.iterows to add in spray data. That was taking so long. Using a dictionary because that can be more efficient.
    spray_dictionary = {}
    for index, row in spray[["Date", "Latitude", "Longitude"]].iterrows():
        try:
            spray_dictionary[row["Date"]].append({"Latitude":row["Latitude"], "Longitude":row["Longitude"]})
        except KeyError:
            spray_dictionary[row["Date"]] = [{"Latitude":row["Latitude"], "Longitude":row["Longitude"]}]
    
    #Rolling up Data so only one record for date and trap available, and also adding spray location based on Date and Location
    train["Date-Trap"] = train["Date"] + "-" + train["Trap"]
    target_remap_dict = {}
    for index, row in train[["Date-Trap", "WnvPresent", "Species", "Date", "Latitude", "Longitude"]].iterrows():
        #Checking for WNV Mosquito Species
        if row["Species"] in ["CULEX PIPIENS/RESTUANS", "CULEX RESTUANS", "CULEX PIPIENS"]:
            wnv_mosquitos = 1
        else:
            wnv_mosquitos = 0
        #Checking if in half mile spray vicinity
        sprayed = 0
        vicenty1 = (row["Latitude"],row["Longitude"])
        try:
            for geocode in spray_dictionary[row["Date"]]:
                vicenty2 = (geocode["Latitude"], geocode["Longitude"])
                if vincenty(vicenty1,vicenty2).miles <= 1:
                    sprayed = 1
                    break
        except KeyError:
            pass
        #Updating Dictionary
        try:
            if target_remap_dict[row["Date-Trap"]]["WnvPresentAdj"] == 0:
                target_remap_dict[row["Date-Trap"]]["WnvPresentAdj"] = row["WnvPresent"]
            if target_remap_dict[row["Date-Trap"]]["WnvMosquito"] == 0:
                target_remap_dict[row["Date-Trap"]]["WnvMosquito"] = wnv_mosquitos
        except KeyError: 
            target_remap_dict[row["Date-Trap"]] = {"WnvPresentAdj": row["WnvPresent"], "WnvMosquito": wnv_mosquitos, "Sprayed": sprayed}
        if (index + 1) % 1000 == 0:
            print index + 1
    target_remap_df = pd.DataFrame.from_dict(target_remap_dict, orient = "index")
    
    #Combining DFs. Using O'Hare Data because Midway does not track certain features
    #Depth and SnowFall likely are bad values because Midway lists them all as M and O'Hare lists them all as 0 (I doubt Chicago had 0 snowfall)
    master_df = train.drop_duplicates(subset = ["Date", "Trap"])
    del master_df['WnvPresent']
    del master_df['Species']
    master_df = pd.merge(master_df, target_remap_df, left_on = "Date-Trap", right_index = True)
    master_df = pd.merge(master_df, weather[weather.Station == 1], left_on = "Date", right_on = "Date", how = "left")
    del master_df["Date-Trap"]
    
    #Converting Date from string to date. The df only has data for odd years which is weird!
    master_df["Date"] = pd.to_datetime(master_df.Date)
    
    return master_df, spray_dictionary

def evaluate_model(y_true, y_predicted):
    a_score = accuracy_score(y_true, y_predicted)
    c_matrix = confusion_matrix(y_true, y_predicted)
    confusion = pd.DataFrame(c_matrix, index=['y_true', 'y_false'], columns=['predicted_true','predicted_false'])
    c_report = classification_report(y_true, y_predicted)
    return a_score, confusion, c_report

train_df = pd.read_csv("assets/train.csv")
weather_df = pd.read_csv("assets/weather.csv")
spray_df = pd.read_csv("assets/spray.csv")
master_df, spray_dictionary = clean_and_transform(train_df, spray_df, weather_df)

#Creating Random Forest
y = master_df.WnvPresentAdj
X = master_df[["Block", "Tmax", "Tmin", "Tavg", "Depart", "DewPoint", "WetBulb", "Heat", "Cool", "Sunrise", "Sunset", "StnPressure", "SeaLevel", "ResultSpeed", "ResultDir", "AvgSpeed"]]
X_train, X_test, y_train, y_test = train_test_split(X, y, stratify = y, test_size=0.3, random_state = 34198)
max_depth = [1,2,3,4,5]
max_features = [0.25, 0.5, 0.75]
forest = RandomForestClassifier(random_state = 98574, n_estimators = 50)
grid = GridSearchCV(forest, param_grid = {'max_depth': max_depth, 'max_features': max_features}, cv = 10, verbose = True, n_jobs = -1)
grid.fit(X_train, y_train)
print grid.score(X_train, y_train)
print grid.score(X_test, y_test)
#Best Score 0.91705354379449089
#Best Params {'max_depth': 5, 'max_features': 0.5}


forest = RandomForestClassifier(random_state = 98574, n_estimators = 50, max_depth = 5, max_features = 0.5)
forest.fit(X_train, y_train)
print forest.score(X_train, y_train)
print forest.score(X_test, y_test)
predictions = forest.predict(X_test)
a_score, confusion, c_report = evaluate_model(y_test, predictions) #Confusion Matrix Printing Out Wrong
print a_score
print confusion
print c_report