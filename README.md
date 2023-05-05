# Song_Feature_Pred
Application to get audio features from Spotify for specific songs, and combine with our own audio dataset to predict the features from spectrograms of the songs

Data Folder:
- ***spotipy_labels.ipynb***: queries Spotify for songs with different feature values ranging from 0.0-1.0 with a step size of 0.005 to make sure that all values are covered. it also queries Spotify for songs with different feature values ranging from 0.0-1.0 with a step size of 0.001 to augment the size of the dataset. Final data collection method was to query 20000 songs across various genres and ensuring that the songs being added for our labels had all the necessary features available.
- ***tracks_data.csv***: data file that includes the songs queried from Spotify with different feature values ranging from 0.0-1.0 with a step size of 0.005 and obtains 7042 songs total. this data was mainly used to grab the audio data
- ***tracks_data_plus.csv***: data file that includes the songs queried from Spotify with different feature values ranging from 0.0-1.0 with a step size of 0.001 and obtains 33,353 songs total
- ***spectrogram_tests.ipynb***: the loop used to extract the spectrogram images from the 30 second preview URLs of the songs from the csv files
- ***tracks_features.csv***: Final folder containing all of our labels and input data including preview urls

Sample Data Folder:
- ***csv files***: includes the csv files for different features of songs such as danceability, energy, speechiness, etc...
- ***dataloader***: an attempt to create a dataloader and cut down the returned tensors, removing white space(turns out manual cropping is needed for this step)
- ***spotipy_test.ipynb***: creates individual dataframes for each of the desired features that we want to predict and collects dataframe for all tracks involved

***model_weights***: Directory containing weights of pretrained models

***song_pred***: environment necessary to run our files locally. It is recommended to load the model weights, labels from

***simple_model.ipynb***: our first attempt at creating a model to feed in the images of the spectrograms extracted from the audio data for a batch of the songs. this simple model uses a convolutional layer and 3 linear layers and uses MSE as the criterion

***mfcc_model.ipynb***: File used to convert all of our audio files into MFCC tensors. Per 30 second audio file, there are 20 coefficients for every frame for 160 frames. First shot at training a CNN model to use these tensors as input for our desired labels. Contains a simple 2D CNN model trained on the tensors, a simple RNN model trained on the mfcc tensors and lastly our multi-task RNN model used for our final architecture. Evaluation metrics provided.

***numerical_model.ipynb***:

***linear_models_combine.ipynb***:

***complex_cnn_mfcc.ipynb***:

***valence_mfcc_model.ipynb***: 

***load_models.ipynb***: Helper file to load numerical regression model and multitask RNN model
