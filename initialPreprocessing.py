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
        X = dataset[feature_combination]
        if processed_X is not None:
            print()
            # deal with combining the artist name with the other info

    elif processed_X is not None:
        X = processed_X
    else:
        X = dataset[[feature]]

    y = dataset['genre_label']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    print("Training sample length: "+str(len(X_train)))
    print("Testing sample length: "+str(len(X_test)))

    return X_train,X_test,y_train,y_test

def top_tracks():
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
    label_encoder = LabelEncoder()
    topg_tracks['genre_label'] = label_encoder.fit_transform(topg_tracks['track_genre_top'])

    return topg_tracks

def genres():
    genre_info = pd.read_csv('fma_metadata/genres.csv')
    return genre_info