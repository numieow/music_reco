import time
import numpy as np
import librosa
import keras
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.discriminant_analysis import StandardScaler
from sklearn.model_selection import GridSearchCV
import tensorflow as tf
from keras.preprocessing import image
from PIL import Image
from tqdm import tqdm
import pickle
#Skip Tensorflow warnings
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

#import KerasClassifier
from scikeras.wrappers import KerasClassifier



INSTRUMENTS_FULL_NAME = {
    'cel':'cello',
    'cla':'clarinet',
    'flu':'flute',
    'gac':'acoustic guitar',
    'gel':'electric guitar',
    'org':'organ',
    'pia':'piano',
    'sax':'saxophone',
    'tru':'trumpet',
    'vio':'violin',
    'voi':'vocals'
}
INSTRUMENTS = list(INSTRUMENTS_FULL_NAME.keys())
TEST_LENGTH = 500

base_path = '../data/'
test_data_path = base_path + 'test_data.csv'
train_data_path = base_path + 'train_data.csv'

# https://medium.com/@nadimkawwa/can-we-guess-musical-instruments-with-machine-learning-afc8790590b8
def feature_extract(file):
    """
    Define function that takes in a file an returns features in an array
    """
    
    #get wave representation : y : waveform, s : sampling rate
    y, sr = librosa.load(file)

    # Print duration of file
    duration = librosa.get_duration(y=y, sr=sr)
    print("Duration: ", duration)
        
    #determine if instrument is harmonic or percussive by comparing means
    y_harmonic, y_percussive = librosa.effects.hpss(y)
    if np.mean(y_harmonic)>np.mean(y_percussive):
        harmonic=1
    else:
        harmonic=0
        
    #Mel-frequency cepstral coefficients (MFCCs)
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
    #temporal averaging
    mfcc=np.mean(mfcc,axis=1)
    
    #get the mel-scaled spectrogram
    spectrogram = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=128,fmax=45000)  
    #temporally average spectrogram
    #spectrogram = np.mean(spectrogram, axis = 1)


    #Specshow the spectrogram
    plt.figure(figsize=(10, 4))
    librosa.display.specshow(librosa.power_to_db(spectrogram, ref=np.max), y_axis='mel', fmax=45000, x_axis='time')
    plt.colorbar(format='%+2.0f dB')
    plt.title('Mel spectrogram')
    plt.tight_layout()
    plt.show()
    
    #compute chroma energy
    chroma = librosa.feature.chroma_cens(y=y, sr=sr)
    #temporally average chroma
    chroma = np.mean(chroma, axis = 1)
    
    #compute spectral contrast
    contrast = librosa.feature.spectral_contrast(y=y, sr=sr)
    contrast = np.mean(contrast, axis= 1)
    
    return [harmonic, mfcc, spectrogram, chroma, contrast]


def spectro_extract(file):
    """
    Define function that takes in a file an returns features in an array
    """
    #TODO : spectro-extract renvoie une liste de 5 elements qui sont toujours le même avec du bruit en plus
    #get wave representation : y : waveform, s : sampling rate
    y, sr = librosa.load(file)
    n_mels = 128
    
    #get the mel-scaled spectrogram
    spectrogram = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=n_mels,fmax=11250)
    spectrogram = librosa.power_to_db(spectrogram, ref=np.max)

    # Specshow the spectrogram
    #plt.figure(figsize=(10, 4))
    #librosa.display.specshow(spectrogram, y_axis='mel', fmax=45000, x_axis='time')
    #plt.colorbar(format='%+2.0f dB')
    #plt.title('Mel spectrogram - BEFORE')
    #plt.tight_layout()
    #plt.show()

    #Normalise the spectrogram by bands of 4 lines
    for i in range(0, n_mels, 4):
        scaler = StandardScaler()
        spectrogram[i:i+4] = scaler.fit_transform(spectrogram[i:i+4])

    # spectrogram_duplicated = spectrogram.copy()
    # np.random.normal(mean, std, output_shape)
    # noise = np.random.normal(0, 25, spectrogram_duplicated.shape).astype(np.uint8)
    # noisy_spectrogram = cv2.add(spectrogram_duplicated, noise)
    # plt.figure()
    # librosa.display.specshow(spectrogram, fmax=45000)
    # plt.axis('off')
    # plt.tight_layout()
    # plt.savefig('temp.png')
    # plt.close()
    # image_pil = Image.open('temp.png')
    #return image.img_to_array(image_pil)

    return spectrogram

#train_df = pd.read_csv(train_data_path)
#train_track_names = pd.unique(train_df['Track Name'])
#oui = spectro_extract('../data/' + train_track_names[0])
#print(oui.shape)

#Take tracks as an input and output multiple tracks
def create_test_tracks(file_name):
    """
    Create multiple tracks from a single file. Each track spans 3 seconds of audio to correspond to the input size of the model.
    Input : string, path to file with extension
    Returns a list of spectrograms.
    """
    y, sr = librosa.load(file_name)
    spectre = spectro_extract(file_name)
    #Get length on y axis
    length = spectre.shape[1]
    target_size = 130
    number_of_subtracks = length // target_size
    total_spectres = [spectre[:, i*target_size:(i+1)*target_size] for i in range(number_of_subtracks)]

    return total_spectres

def one_hot_encode(y):
    for i in range(len(y)):
        y[i] = [1 if instr in y[i] else 0 for instr in INSTRUMENTS]
    return y

def create_model():
    """
    Creates and returns a model
    """
    model = keras.models.Sequential(
    [
        keras.layers.Input(shape=(128, 130, 1)),
        keras.layers.Conv2D(16, (7, 7), activation='relu'),
        keras.layers.Conv2D(32, (5, 5), activation='relu'),
        keras.layers.MaxPooling2D(),
        keras.layers.Dropout(0.25),
        keras.layers.Conv2D(64, (3, 3), activation='relu'),
        keras.layers.MaxPooling2D(),
        keras.layers.Conv2D(128, (3, 3), activation='relu'),
        keras.layers.MaxPooling2D(),
        keras.layers.Dropout(0.5),
        keras.layers.Conv2D(256, (3, 3), activation='relu'),
        keras.layers.Conv2D(512, (3, 3), activation='relu'),
        keras.layers.MaxPooling2D(),
        keras.layers.Flatten(),
        keras.layers.Dropout(0.5),
        keras.layers.Dense(100, activation='relu'),
        keras.layers.Dropout(0.75),
        keras.layers.Dense(30, activation='relu'),
        keras.layers.Dropout(0.75),
        keras.layers.Dense(len(INSTRUMENTS_FULL_NAME), activation='sigmoid')
    ]
)

    #model.summary()
    optimizer = keras.optimizers.Adam(learning_rate=0.00001)
    model.compile(loss='categorical_crossentropy', optimizer=optimizer, metrics=['accuracy'])
    return model

def create_optimal_model():
    model = keras.models.Sequential(
        [
            keras.layers.Input(shape=(128, 130, 1)),
            keras.layers.Conv2D(32, (3, 3), activation='relu'),
            keras.layers.MaxPooling2D(),
            keras.layers.Conv2D(128, (3, 3), activation='relu'),
            keras.layers.MaxPooling2D(),
            keras.layers.Conv2D(128, (3, 3), activation='relu'),
            keras.layers.MaxPooling2D(),
            keras.layers.Conv2D(128, (3, 3), activation='relu'),
            keras.layers.MaxPooling2D(),
            keras.layers.Flatten(),
            keras.layers.Dense(256, activation='relu'),
            keras.layers.Dense(len(INSTRUMENTS_FULL_NAME), activation='sigmoid')
        ]
    )

    optimizer = keras.optimizers.Adam(learning_rate=0.00001)
    model.compile(loss='categorical_crossentropy', optimizer=optimizer, metrics=['accuracy'])
    return model

def display_cv_results(search_results):
    print('Best score = {:.4f} using {}'.format(search_results.best_score_, search_results.best_params_))
    means = search_results.cv_results_['mean_test_score']
    stds = search_results.cv_results_['std_test_score']
    params = search_results.cv_results_['params']
    for mean, stdev, param in zip(means, stds, params):
        print('mean test accuracy +/- std = {:.4f} +/- {:.4f} with: {}'.format(mean, stdev, param))  
        # print keys of the cv_results_ dictionary
        print(search_results.cv_results_.keys())

# explicit function to normalize array
def normalize_2d(matrix):
    norm = np.linalg.norm(matrix)
    matrix = matrix/norm  # normalized matrix
    return matrix


"""print("### TRAIN DATA ###")
train_df = pd.read_csv(train_data_path)
train_track_names = pd.unique(train_df['Track Name'])

x_train = [spectro_extract('../data/' + file) for file in tqdm(train_track_names)]

for i in tqdm(range(len(x_train))):
    x_train[i] = tf.convert_to_tensor(x_train[i], dtype=tf.float32)

y_train = [train_df[train_df['Track Name'] == name]['Instrument'].values.tolist() for name in tqdm(train_track_names)]
y_train = one_hot_encode(y_train)

for i in tqdm(range(len(y_train))):
    y_train[i] = tf.convert_to_tensor(y_train[i], dtype=tf.float32)

x_train = np.array(x_train)
y_train = np.array(y_train)


with open("x_train_norm", "wb") as fp:   #Pickling
    pickle.dump(x_train, fp)
with open("y_train_norm", "wb") as fp:   #Pickling
    pickle.dump(y_train, fp)


print("### TEST DATA ###")
test_df = pd.read_csv(test_data_path)
test_track_names = pd.unique(test_df['Track Name'])

x_test = [create_test_tracks('../data/' + file + '.wav') for file in tqdm(test_track_names)]
for i in tqdm(range(len(x_test))):
    x_test[i] = tf.convert_to_tensor(x_test[i], dtype=tf.float32)

y_test = [test_df[test_df['Track Name'] == name]['Instrument'].values.tolist() for name in tqdm(test_track_names)]
y_test = one_hot_encode(y_test)
for i in tqdm(range(len(y_test))):
    y_test[i] = tf.convert_to_tensor(y_test[i], dtype=tf.float32)

new_x_test = []
new_y_test = []
print("### REPLACEMENT ###")
for i in tqdm(range(len(x_test))):
    curr_list = x_test[i]
    for elem in curr_list:
        new_x_test.append(elem)
        new_y_test.append(y_test[i])

x_test = np.array(new_x_test)
y_test = np.array(new_y_test)

with open("x_test_norm", "wb") as fp:   #Pickling
    pickle.dump(x_test, fp)
with open("y_test_norm", "wb") as fp:   #Pickling
    pickle.dump(y_test, fp)"""

# Retrieve data
with open("x_train_norm", "rb") as fp:   # Unpickling
    x_train = pickle.load(fp)

with open("y_train_norm", "rb") as fp:   # Unpickling
    y_train = pickle.load(fp)

with open("x_test_norm", "rb") as fp:   # Unpickling
    x_test = pickle.load(fp)
    
with open("y_test_norm", "rb") as fp:   # Unpickling
    y_test = pickle.load(fp)

"""
# Snippet to show a spectrogram 
plt.figure(figsize=(10, 4))
librosa.display.specshow(x_train[0], y_axis='mel', fmax=11250, x_axis='time')
plt.colorbar(format='%+2.0f dB')
plt.title('Mel spectrogram')
plt.tight_layout()
plt.show()"""


#Start a timer
start = time.time()

# create model
model = KerasClassifier(model=create_model, verbose=1)
# define parameters and values for grid search 
n_cv = 3
n_epochs_cv = 50

param_grid = {
    'batch_size': [8, 16, 32, 64],
    'epochs': [n_epochs_cv],
    'validation_split': [0.1, 0.2, 0.3, 0.4, 0.5]
}

"""
grid = GridSearchCV(estimator=model, param_grid=param_grid, n_jobs=1, cv=n_cv, error_score='raise')
grid_result = grid.fit(x_train, y_train) 
print('time for grid search = {:.0f} sec'.format(time.time()-start))
display_cv_results(grid_result)"""


"""
# reload best model
mlp = grid_result.best_estimator_ 
checkpoint_path = "audio_model.keras"

model_checkpoint_callback = keras.callbacks.ModelCheckpoint(
    filepath=checkpoint_path,
    monitor='val_loss',
    mode='min',
    save_best_only=True,
    verbose=1
)
# retrain best model on the full training set 
history = mlp.fit(x_train, y_train, batch_size=16, epochs=10, validation_split=0.2, callbacks=[model_checkpoint_callback])
"""



checkpoint_path = "audio_model.keras"

model_checkpoint_callback = keras.callbacks.ModelCheckpoint(
    filepath=checkpoint_path,
    monitor='val_loss',
    mode='min',
    save_best_only=True,
    verbose=1
)

model = create_model()
print(model.summary())

#Params "optimaux" : bs 32, epochs 100, validation_split 0.3
hist = model.fit(x_train, y_train, batch_size=8, epochs=150, validation_split=0.4, callbacks=[model_checkpoint_callback])

#Load model
loaded = keras.models.load_model("audio_model.keras")

score = loaded.evaluate(x_test, y_test, verbose=0)
print("Test loss : ", score[0])
print("Test accuracy", score[1])
plt.plot(hist.history['accuracy'])
plt.plot(hist.history['val_accuracy'])
plt.title('Model Accuracy')
plt.ylabel('Accuracy')
plt.xlabel('Epochs')
plt.legend(['train', 'val'], loc='upper left')
plt.show()
plt.plot(hist.history['loss'])
plt.plot(hist.history['val_loss'])
plt.title('Model Loss')
plt.ylabel('Loss')
plt.xlabel('Epochs')
plt.legend(['train', 'val'], loc='upper left')
plt.show()