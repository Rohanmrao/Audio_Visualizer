<<<<<<< HEAD
# imoprt libraries
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
#%matplotlib inline

from glob import glob # allows us to list all files to a directory
import IPython
import IPython.display as ipd # to play the Audio Files

import librosa # main package for working with Audio Data
import librosa.display

from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score , confusion_matrix , ConfusionMatrixDisplay , classification_report

from sklearn.model_selection import KFold
from sklearn.model_selection import GridSearchCV

# Make a list of all the wav files in the dataset and store them in a variable
audio_files = glob("C:/Users/Rohan Mahesh Rao/Documents/PES1UG20EC156/Sem 6/ML/project/gtzan/Data/genres_original/*/*.wav")
audio_files = [path.replace('//', '/') for path in audio_files]
print(type(audio_files))

# load the audio file and show raw data and sample rate
y, sr = librosa.load(audio_files[0])
print("Y is a numpy array:", y)
print("Shape of Y:", y.shape)
print("Sample Rate:", sr)
def visualise_song(filename):
    
    y, sr = librosa.load(filename, sr=None)

    # turn raw data array to pd series and plot the audio example
    pd.Series(y).plot(figsize=(8,2), title="Raw Audio Example", color='green');

    # Use STFT on raw audio data
    D = librosa.stft(y)
    # convert from aplitude to decibel values by taking the absolute value of D in reference what the max value would be
    S_db = librosa.amplitude_to_db(np.abs(D), ref=np.max)
    # see the shape of transformed data
    print("New shape of transformed data", S_db.shape)
    
    # plot transformed data as spectogram
    fig, ax = plt.subplots(figsize=(3,3))
    img = librosa.display.specshow(S_db, x_axis='time', y_axis='log', ax=ax)
    ax.set_title('Spectogram Example', fontsize=20)
    fig.colorbar(img, ax=ax, format=f'%0.2f');

def get_mfcc(y,sr):

    # Mel-frequency cepstral coefficients (MFCCs)
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=20)
    mfcc_mean_vars = []
 
    for i in range (20):

        mfcc_mean_vars.append(np.mean(mfcc[i]))
        mfcc_mean_vars.append(np.var(mfcc[i]))

    mfcc_mean_vars = np.array(mfcc_mean_vars)

    mfcc_mean_vars = mfcc_mean_vars.tolist()

    return mfcc_mean_vars
#### The audio feature extraction fucntion ##########

## inputs must be of type string

import librosa
import numpy as np

def extract_features(filename):

    y, sr = librosa.load(filename, sr=None)

    visualise_song(filename)

    mfcc_values = get_mfcc(y,sr) #1d

    # Spectral features
    spectral_centroid = librosa.feature.spectral_centroid(y=y, sr=sr)
    spectral_bandwidth = librosa.feature.spectral_bandwidth(y=y, sr=sr)
    spectral_rolloff = librosa.feature.spectral_rolloff(y=y, sr=sr)

    c_mean = np.mean(np.mean(spectral_centroid))
    b_mean = np.mean(np.mean(spectral_bandwidth))
    r_mean = np.mean(np.mean(spectral_rolloff))
    
    # Chroma features
    chroma_stft = librosa.feature.chroma_stft(y=y, sr=sr)
    chroma_stft_mean = np.mean(np.mean(chroma_stft))
    chroma_stft_var = np.var(np.var(chroma_stft))
    
    # Root-mean-square (RMS) energy
    rmse = librosa.feature.rms(y=y)
    rms_mean = np.mean(np.mean(rmse))
    rms_var = np.var(np.var(rmse))
    
    # Zero-crossing rate
    zcr = librosa.feature.zero_crossing_rate(y)
    zcr_mean = np.mean(np.mean(zcr))
    zcr_var = np.var(np.var(zcr))
    
    # Harmony features
    harmonic_percussive = librosa.effects.hpss(y)
    harmony = librosa.feature.tonnetz(y=harmonic_percussive[0], sr=sr)
    harmony_mean = np.mean(np.mean(harmony))
    harmony_var = np.var(np.var(harmony))
    
    # Perceived loudness
    perceived_loudness = librosa.feature.spectral_flatness(y=y)
    perceived_loudness_mean = np.mean(np.mean(perceived_loudness))
    perceived_loudness_var = np.var(np.var(perceived_loudness))
    
    # Tempo
    tempo, _ = librosa.beat.beat_track(y=y, sr=sr) # 0d 
    
    features = []
    features.append([chroma_stft_mean,
                                rms_mean,
                                c_mean,
                                b_mean,
                                r_mean,
                                zcr_mean,
                                harmony_mean,
                                perceived_loudness_mean,tempo])

    features_ls = features[0]

    features_ls.extend(mfcc_values)
   
 
    return features_ls

# load csv file
df = pd.read_csv("C:/Users/Rohan Mahesh Rao/Documents/PES1UG20EC156/Sem 6/ML/project/gtzan/Data/features_3_sec.csv")
df.head() # first 5 entries
df.shape # see the shape of df
# df.info() # infos about the samples, features and datatypes
#df.isnull().sum() # checking for missing values
# drop filename column and show new df first 5 entries
df = df.drop(labels=['length','filename','chroma_stft_var','rms_var','spectral_centroid_var','spectral_bandwidth_var','rolloff_var','zero_crossing_rate_var','harmony_var','perceptr_var'],axis=1)
df.head()
# import labelencoder and scaler
from sklearn.preprocessing import LabelEncoder , OneHotEncoder
from sklearn.preprocessing import StandardScaler
encoder = LabelEncoder()
scaler = StandardScaler()

data = df.iloc[:, :-1] # obtain metadata
labels = df.iloc[:, -1] # get labels column
labels.to_frame() # change datatype to pandas dataframe

print(labels)
# assign x and y, scale x and encode y
x = np.array(data, dtype = float)
x = scaler.fit_transform(data)

#### IMPORTANT #####

print (type(x))

########### the music features must also be of the type numpy.ndarray #################
y = encoder.fit_transform(labels)

print(y)

x.shape, y.shape
# split data to train and test data
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2)

## verify the type of x_train and x_test
print(type(x_train),type(x_test))

x_train.shape, x_test.shape, y_train.shape, y_test.shape
def cross_val(classifier_rf, K, metadata, label, title, return_clf = False):
    # scores is used to give average of accuracy
    scores = []
    cv = KFold(n_splits=K)
    
    # K fold analysis, used for spliting the data into k batches
    for train_index, test_index in cv.split(metadata):
      
        X_train, y_train = metadata[train_index], label[train_index]
        X_test, y_test = metadata[test_index], label[test_index]

        classifier_rf.fit(X_train, y_train)
        scores.append(classifier_rf.score(X_test, y_test))
    
    # Display the average score
    print(title + " Cross-Validation Accuracy Score: ", round(np.mean(scores), 2))
    
    # returns the classifier if needed
    if(return_clf == True):
        return classifier_rf

def display_confusionMatrix(classifier_rf, X, y, title):
  cm = confusion_matrix(y, classifier_rf.predict(X), labels=classifier_rf.classes_)
  normed_c = (cm.T / cm.astype(np.float).sum(axis=1)).T
  disp = ConfusionMatrixDisplay(confusion_matrix=normed_c,
                                display_labels=classifier_rf.classes_)
  disp.plot(
      cmap=plt.cm.Blues
      )
  disp.ax_.set_title(title + " Confusion Matrix:")
XGB=XGBClassifier()
XGB.fit(x_train,y_train)
print(x_train.shape)
print(x_test.shape)
y_pred = XGB.predict(x_test)

print("Model Accuracy : ",accuracy_score(y_test,y_pred))

cross_val(XGB, 5, x_test, y_test, title="Cross Val", return_clf = False)

print(classification_report(y_test, XGB.predict(x_test)))

display_confusionMatrix(XGB, x_test, y_test, title = "Conf matrix")
testpath = "C:/Users/Rohan Mahesh Rao/Documents/PES1UG20EC156/Sem 6/ML/project/gtzan/Data/genres_original/jazz/jazz.00003.wav"
# Play the first Audio file
ipd.Audio(testpath)
   
original_labels = encoder.inverse_transform(y)
print(original_labels)
vals = extract_features(testpath)

extracted_tempo = np.round(vals[8])

label_ls = ["classical", "jazz", "metal"]

pred = XGB.predict([vals])

final_data = [label_ls[pred[0]],extracted_tempo]
=======
# imoprt libraries
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
#%matplotlib inline

from glob import glob # allows us to list all files to a directory
import IPython
import IPython.display as ipd # to play the Audio Files

import librosa # main package for working with Audio Data
import librosa.display

from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score , confusion_matrix , ConfusionMatrixDisplay , classification_report

from sklearn.model_selection import KFold
from sklearn.model_selection import GridSearchCV

# Make a list of all the wav files in the dataset and store them in a variable
audio_files = glob("C:/Users/Rohan Mahesh Rao/Documents/PES1UG20EC156/Sem 6/ML/project/gtzan/Data/genres_original/*/*.wav")
audio_files = [path.replace('//', '/') for path in audio_files]
print(type(audio_files))

# load the audio file and show raw data and sample rate
y, sr = librosa.load(audio_files[0])
print("Y is a numpy array:", y)
print("Shape of Y:", y.shape)
print("Sample Rate:", sr)
def visualise_song(filename):
    
    y, sr = librosa.load(filename, sr=None)

    # turn raw data array to pd series and plot the audio example
    pd.Series(y).plot(figsize=(8,2), title="Raw Audio Example", color='green');

    # Use STFT on raw audio data
    D = librosa.stft(y)
    # convert from aplitude to decibel values by taking the absolute value of D in reference what the max value would be
    S_db = librosa.amplitude_to_db(np.abs(D), ref=np.max)
    # see the shape of transformed data
    print("New shape of transformed data", S_db.shape)
    
    # plot transformed data as spectogram
    fig, ax = plt.subplots(figsize=(3,3))
    img = librosa.display.specshow(S_db, x_axis='time', y_axis='log', ax=ax)
    ax.set_title('Spectogram Example', fontsize=20)
    fig.colorbar(img, ax=ax, format=f'%0.2f');

def get_mfcc(y,sr):

    # Mel-frequency cepstral coefficients (MFCCs)
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=20)
    mfcc_mean_vars = []
 
    for i in range (20):

        mfcc_mean_vars.append(np.mean(mfcc[i]))
        mfcc_mean_vars.append(np.var(mfcc[i]))

    mfcc_mean_vars = np.array(mfcc_mean_vars)

    mfcc_mean_vars = mfcc_mean_vars.tolist()

    return mfcc_mean_vars
#### The audio feature extraction fucntion ##########

## inputs must be of type string

import librosa
import numpy as np

def extract_features(filename):

    y, sr = librosa.load(filename, sr=None)

    visualise_song(filename)

    mfcc_values = get_mfcc(y,sr) #1d

    # Spectral features
    spectral_centroid = librosa.feature.spectral_centroid(y=y, sr=sr)
    spectral_bandwidth = librosa.feature.spectral_bandwidth(y=y, sr=sr)
    spectral_rolloff = librosa.feature.spectral_rolloff(y=y, sr=sr)

    c_mean = np.mean(np.mean(spectral_centroid))
    b_mean = np.mean(np.mean(spectral_bandwidth))
    r_mean = np.mean(np.mean(spectral_rolloff))
    
    # Chroma features
    chroma_stft = librosa.feature.chroma_stft(y=y, sr=sr)
    chroma_stft_mean = np.mean(np.mean(chroma_stft))
    chroma_stft_var = np.var(np.var(chroma_stft))
    
    # Root-mean-square (RMS) energy
    rmse = librosa.feature.rms(y=y)
    rms_mean = np.mean(np.mean(rmse))
    rms_var = np.var(np.var(rmse))
    
    # Zero-crossing rate
    zcr = librosa.feature.zero_crossing_rate(y)
    zcr_mean = np.mean(np.mean(zcr))
    zcr_var = np.var(np.var(zcr))
    
    # Harmony features
    harmonic_percussive = librosa.effects.hpss(y)
    harmony = librosa.feature.tonnetz(y=harmonic_percussive[0], sr=sr)
    harmony_mean = np.mean(np.mean(harmony))
    harmony_var = np.var(np.var(harmony))
    
    # Perceived loudness
    perceived_loudness = librosa.feature.spectral_flatness(y=y)
    perceived_loudness_mean = np.mean(np.mean(perceived_loudness))
    perceived_loudness_var = np.var(np.var(perceived_loudness))
    
    # Tempo
    tempo, _ = librosa.beat.beat_track(y=y, sr=sr) # 0d 
    
    features = []
    features.append([chroma_stft_mean,
                                rms_mean,
                                c_mean,
                                b_mean,
                                r_mean,
                                zcr_mean,
                                harmony_mean,
                                perceived_loudness_mean,tempo])

    features_ls = features[0]

    features_ls.extend(mfcc_values)
   
 
    return features_ls

# load csv file
df = pd.read_csv("C:/Users/Rohan Mahesh Rao/Documents/PES1UG20EC156/Sem 6/ML/project/gtzan/Data/features_3_sec.csv")
df.head() # first 5 entries
df.shape # see the shape of df
# df.info() # infos about the samples, features and datatypes
#df.isnull().sum() # checking for missing values
# drop filename column and show new df first 5 entries
df = df.drop(labels=['length','filename','chroma_stft_var','rms_var','spectral_centroid_var','spectral_bandwidth_var','rolloff_var','zero_crossing_rate_var','harmony_var','perceptr_var'],axis=1)
df.head()
# import labelencoder and scaler
from sklearn.preprocessing import LabelEncoder , OneHotEncoder
from sklearn.preprocessing import StandardScaler
encoder = LabelEncoder()
scaler = StandardScaler()

data = df.iloc[:, :-1] # obtain metadata
labels = df.iloc[:, -1] # get labels column
labels.to_frame() # change datatype to pandas dataframe

print(labels)
# assign x and y, scale x and encode y
x = np.array(data, dtype = float)
x = scaler.fit_transform(data)

#### IMPORTANT #####

print (type(x))

########### the music features must also be of the type numpy.ndarray #################
y = encoder.fit_transform(labels)

print(y)

x.shape, y.shape
# split data to train and test data
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2)

## verify the type of x_train and x_test
print(type(x_train),type(x_test))

x_train.shape, x_test.shape, y_train.shape, y_test.shape
def cross_val(classifier_rf, K, metadata, label, title, return_clf = False):
    # scores is used to give average of accuracy
    scores = []
    cv = KFold(n_splits=K)
    
    # K fold analysis, used for spliting the data into k batches
    for train_index, test_index in cv.split(metadata):
      
        X_train, y_train = metadata[train_index], label[train_index]
        X_test, y_test = metadata[test_index], label[test_index]

        classifier_rf.fit(X_train, y_train)
        scores.append(classifier_rf.score(X_test, y_test))
    
    # Display the average score
    print(title + " Cross-Validation Accuracy Score: ", round(np.mean(scores), 2))
    
    # returns the classifier if needed
    if(return_clf == True):
        return classifier_rf

def display_confusionMatrix(classifier_rf, X, y, title):
  cm = confusion_matrix(y, classifier_rf.predict(X), labels=classifier_rf.classes_)
  normed_c = (cm.T / cm.astype(np.float).sum(axis=1)).T
  disp = ConfusionMatrixDisplay(confusion_matrix=normed_c,
                                display_labels=classifier_rf.classes_)
  disp.plot(
      cmap=plt.cm.Blues
      )
  disp.ax_.set_title(title + " Confusion Matrix:")
XGB=XGBClassifier()
XGB.fit(x_train,y_train)
print(x_train.shape)
print(x_test.shape)
y_pred = XGB.predict(x_test)

print("Model Accuracy : ",accuracy_score(y_test,y_pred))

cross_val(XGB, 5, x_test, y_test, title="Cross Val", return_clf = False)

print(classification_report(y_test, XGB.predict(x_test)))

display_confusionMatrix(XGB, x_test, y_test, title = "Conf matrix")
testpath = "C:/Users/Rohan Mahesh Rao/Documents/PES1UG20EC156/Sem 6/ML/project/gtzan/Data/genres_original/jazz/jazz.00003.wav"
# Play the first Audio file
ipd.Audio(testpath)
   
original_labels = encoder.inverse_transform(y)
print(original_labels)
vals = extract_features(testpath)

extracted_tempo = np.round(vals[8])

label_ls = ["classical", "jazz", "metal"]

pred = XGB.predict([vals])

final_data = [label_ls[pred[0]],extracted_tempo]
>>>>>>> origin/main
print(final_data)