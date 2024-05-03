import pandas as pd
import matplotlib.pyplot as plt
from initialPreprocessing import top_tracks
import os
import numpy as np
import matplotlib.pyplot as plt
import math

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
        print("with date"+str(len(topg_tracks)))

    topg_tracks['genre_label'] = topg_tracks['track_genre_top']

    return topg_tracks

def plot_date_recorded_distribution(data, feature='track_date_recorded', genre_column='genre_label'):
    # Convert 'track_date_recorded' column to datetime
    data[feature] = pd.to_datetime(data[feature])

    # Group data by genre
    grouped_data = data.groupby(genre_column)

    # Plot the distribution of 'date_recorded' for each genre
    for genre, group in grouped_data:
        plt.figure(figsize=(10, 6))
        plt.hist(group[feature], bins=30, alpha=0.7, label=genre)
        plt.title(f'Distribution of {feature} for {genre} genre')
        plt.xlabel(feature)
        plt.ylabel('Frequency')
        plt.legend()
        plt.show()

if __name__ == "__main__":
    sample = top_tracks(True)

    # Call the function to plot the distribution
    plot_date_recorded_distribution(sample)
