import os
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'

# python 00_Preprocessing.py
# execfile( '00_Preprocessing.py' )

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pandas.plotting import scatter_matrix


# Spotify data preprocessing


# read dataset
print 'Read data'
print

filenames = [ "data_reggaeton.csv", "data_todotipo.csv", "data_test.csv" ]
#filenames = [ "data_reggaeton.csv", "data_todotipo.csv" ]
df_list = []

for k, filename in zip( range( len(filenames) ), filenames ):
    df_list.append( pd.read_csv( filename ) )
    df = df_list[k]

    if 1:
        print 'Data summary ' + filename
        print df.shape
        print
        
        print df.dtypes
        print
        
        #print df.head()
        #print df.tail()
        #print

    if 0:
        # full set reference
        #subset = [ 'acousticness', 'danceability', 'duration', 'energy', 'id_new', 'instrumentalness', 'key', 'liveness', 'loudness', 'mode', 'music type', 'popularity', 'speechiness', 'tempo', 'time_signature', 'valence' ]
        fieldNames = df.columns
        print 'Fields', fieldNames
        print

    if 1:
        # check for empty values and count them (missing data)
        print 'Empty records quantity'
        print df[df.isnull().any(axis=1)].shape
        print

        #print df.isnull().sum(axis=1)
        print df.isnull().sum(axis=0)
        print


if 1:
    # targeted preprocessing 1 (reggaeton)
    df = df_list[0]
    
    # normalize dataframe field quantity
    df['time_signature'] = 0.0

    # add music type flag
    df['music type'] = 'reggaeton'

    # float to int conversion
    newTypes = { 'duration': np.float, 'id_new': np.float, 'key': np.float, 'mode': np.float, 'popularity': np.float }
    df = df.astype( newTypes )

    df_list[0] = df

    # check dataframe
    print 'Preprocessing 1'
    print df_list[0].shape
    print


    # targeted preprocessing 2 (otros)
    df = df_list[1]
    
    # float to int conversion
    newTypes = { 'id_new': np.float }
    df = df.astype( newTypes )
    
    # add music type flag
    df['music type'] = 'otros'

    # check contents of empty values
    print df[df.isnull().any(axis=1)]
    print

    # remove empty records
    df_list[1] = df[df.isnull().any(axis=1) != True]
    
    # check dataframe
    print 'Preprocessing 2'
    print df_list[1].shape
    print


    # targeted preprocessing 3 (test dataset)
    df = df_list[2]
    
    # add music type flag
    df['music type'] = ''
    
    # float to int conversion
    newTypes = { 'duration': np.float, 'id_new': np.float, 'key': np.float, 'mode': np.float, 'popularity': np.float, 'time_signature': np.float }
    df = df.astype( newTypes )
    
    df_list[2] = df
    
    # check dataframe
    print 'Preprocessing 3'
    print df_list[2].shape
    print


if 1:
    print 'Save CSV file'
    print
    
    # concatenate training data
    df_concat = pd.concat( [df_list[0], df_list[1]] )
    
    # check final size
    print 'Save train dataset'
    print df_concat.shape
    print
    
    # save train dataset
    df_concat.to_csv( 'data_train.csv' )


    # check final size
    print 'Save test dataset'
    print df_list[2].shape
    print

    # save test dataset
    df_list[2].to_csv( 'data_test_proc.csv' )


if 1:
    # plot histograms to check data distribution
    print 'Data histograms'
    print
    
    for df in df_list:
        # change scale for better visualization
        df['duration'] = df['duration'] / ( 1000.0 * 60.0 )
        
        # keep relevant variables for better visualization
        subset = [ 'acousticness', 'danceability', 'duration', 'energy', 'instrumentalness', 'key', 'liveness', 'loudness', 'mode', 'popularity', 'speechiness', 'tempo', 'valence' ]
        
        df = df[subset]
        df.hist( figsize=(13, 8), xlabelsize=8, ylabelsize=8, color='orange' )

    plt.show()


if 0:
    # plot scatters to check linear relationship
    print 'Data scatters'
    print
    
    for df in df_list:
        # change scale for better visualization
        df['duration'] = df['duration'] / ( 1000.0 * 60.0 )
        
        # keep relevant variables for better visualization
        subset = [ 'acousticness', 'danceability', 'duration', 'energy', 'instrumentalness', 'key', 'liveness', 'loudness', 'mode', 'popularity', 'speechiness', 'tempo', 'valence' ]
    
        df = df[subset]
        scatter_matrix( df, alpha=0.2, figsize=(13, 8) )

    plt.show()



