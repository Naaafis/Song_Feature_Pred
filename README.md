# Song_Feature_Pred
Application to get audio features from Spotify for specific songs, and combine with our own audio dataset to predict the features from spectrograms of the songs

Data Folder:
- ***spotipy_labels.ipynb***: queries Spotify for songs with different feature values ranging from 0.0-1.0 with a step size of 0.005 to make sure that all values are covered. it also queries Spotify for songs with different feature values ranging from 0.0-1.0 with a step size of 0.001 to augment the size of the dataset
- ***tracks_data.csv***: data file that includes the songs queried from Spotify with different feature values ranging from 0.0-1.0 with a step size of 0.005 and obtains 7042 songs total. this data was mainly used to grab the audio data
- ***tracks_data_plus.csv***: data file that includes the songs queried from Spotify with different feature values ranging from 0.0-1.0 with a step size of 0.001 and obtains 33,353 songs total

Sample Data Folder:
- ***csv files***: includes the csv files for different features of songs such as danceability, energy, speechiness, etc...
- ***spotipy_test.ipynb***: creates individual dataframes for each of the desired features that we want to predict and collects dataframe for all tracks involved

***song_pred***: environment necessary to run our files in.

***dataloader***: an attempt to createa a dataloader and cut down the returned tensors, removing white space(turns out manual cropping is needed for this step)

***simple_model.ipynb***: our first attempt at creating a model to feed in the images of the spectrograms extracted from the audio data for a batch of the songs. this simple model uses a convolutional layer and 3 linear layers and uses MSE as the criterion

***spectrogram_tests.ipynb***: the loop used to extract the spectrogram images from the 30 second preview URLs of the songs from the csv files
