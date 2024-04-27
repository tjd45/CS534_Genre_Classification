import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import math

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

def gen_Train_and_Test(data,feature, subset,processed_X=None,feature_combination=[]):
    if(subset != 0):
        dataset = data.sample(n=subset,random_state=42)
    else:
        dataset = data
    
    if len(feature_combination)>0:
        if processed_X is not None:
            X1 = dataset[feature_combination]
            X1.reset_index(drop=True,inplace=True)
            X2 = pd.DataFrame(processed_X)
            X2.reset_index(drop=True,inplace=True)
            

            X = pd.concat([X1, X2], axis=1)
            # bc the vectorised representation of the track names give the column titles as integers had to change this
            X.columns = X.columns.astype(str)
        
        else:
            X = dataset[feature_combination]
  
    elif processed_X is not None:
        X = processed_X
    
    else:
        X = dataset[[feature]]
    

    y = dataset['genre_label']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    print("Training sample length: "+str(len(X_train)))
    print("Testing sample length: "+str(len(X_test)))

    return X_train,X_test,y_train,y_test

def top_tracks(daterecorded=False):
    track_headers = pd.read_csv('fma_metadata/tracks.csv',nrows=3, header=None)
    new_track_headers = []

    for col in track_headers:
        if not isinstance(track_headers[col].iloc[0],str) :
            new_track_headers.append(track_headers[col].iloc[2])
        else:
            new_track_headers.append(track_headers[col].iloc[0]+"_"+track_headers[col].iloc[1])

    tracks = pd.read_csv('fma_metadata/tracks.csv',skiprows=[0,1,2], header=None)
    tracks.columns = new_track_headers
    genre_info = pd.read_csv('fma_metadata/genres.csv')
    topg_tracks = tracks.dropna(subset=['track_genre_top']).copy()
    topg_tracks = topg_tracks.dropna(subset=['track_title']).copy()

    if daterecorded:
        # Ensure the 'track_date_recorded' column is a datetime object
        topg_tracks['track_date_recorded'] = pd.to_datetime(topg_tracks['track_date_recorded'])

        # Calculate the number of days since the first date in the dataset
        min_date =  topg_tracks['track_date_recorded'].min()
        topg_tracks['days_since_first'] = (topg_tracks['track_date_recorded'] - min_date).dt.days
        topg_tracks = topg_tracks.dropna(subset=['track_date_recorded']).copy()


    label_encoder = LabelEncoder()
    topg_tracks['genre_label'] = label_encoder.fit_transform(topg_tracks['track_genre_top'])

    return topg_tracks

def top_echonest_tracks(daterecorded=False):
    topg_tracks = top_tracks()

    echonest_headers = pd.read_csv('fma_metadata/echonest.csv',nrows=4, header=None)
    new_echonest_headers = []

    for col in echonest_headers:
        if not isinstance(echonest_headers[col].iloc[0],str) :
            new_echonest_headers.append(echonest_headers[col].iloc[3])
        else:
            new_echonest_headers.append(echonest_headers[col].iloc[0]+"_"+echonest_headers[col].iloc[2])

    # print(echonest_headers)
    # print(new_echonest_headers)

    echonest = pd.read_csv('fma_metadata/echonest.csv',skiprows=[0,1,2,3],header=None)
    echonest.columns = new_echonest_headers


    topg_echo_merged = pd.merge(topg_tracks,echonest,on='track_id',how='inner')

    get_genre_info(topg_echo_merged)

    if daterecorded:
        # Ensure the 'track_date_recorded' column is a datetime object
        topg_echo_merged['track_date_recorded'] = pd.to_datetime(topg_echo_merged['track_date_recorded'])

        # Calculate the number of days since the first date in the dataset
        min_date =  topg_echo_merged['track_date_recorded'].min()
        topg_echo_merged['days_since_first'] = (topg_echo_merged['track_date_recorded'] - min_date).dt.days
        topg_echo_merged = topg_echo_merged.dropna(subset=['track_date_recorded']).copy()

    return topg_echo_merged

def top_n_genre_tracks(n):
    track_headers = pd.read_csv('fma_metadata/tracks.csv',nrows=3, header=None)
    new_track_headers = []

    for col in track_headers:
        if not isinstance(track_headers[col].iloc[0],str) :
            new_track_headers.append(track_headers[col].iloc[2])
        else:
            new_track_headers.append(track_headers[col].iloc[0]+"_"+track_headers[col].iloc[1])

    tracks = pd.read_csv('fma_metadata/tracks.csv',skiprows=[0,1,2], header=None)
    tracks.columns = new_track_headers

    topg_tracks = tracks.dropna(subset=['track_genre_top']).copy()
    topg_tracks = topg_tracks.dropna(subset=['track_title']).copy()

    top_genres=get_genre_info(topg_tracks,False)

    top_n_genres = [genre for genre, _ in top_genres.items()][:n]

    topg_tracks = topg_tracks[topg_tracks['track_genre_top'].isin(top_n_genres)].copy()

    label_encoder = LabelEncoder()
    topg_tracks['genre_label'] = label_encoder.fit_transform(topg_tracks['track_genre_top'])
    
    return topg_tracks

def get_genre_info(tracks, output=True):
    genre_counts = tracks['track_genre_top'].value_counts()

    total_tracks = len(tracks)
    runningCount = 0
    if(output):
        for genre, count in genre_counts.items():
            runningCount += count
            percentage = (count / total_tracks) * 100
            runningPct = (runningCount / total_tracks) * 100
            print(f"{genre}: {percentage:.2f}% ({count} tracks) - {runningPct:.2f}% total")
    
    return genre_counts


def genres():
    genre_info = pd.read_csv('fma_metadata/genres.csv')
    return genre_info

# top_echonest_tracks()