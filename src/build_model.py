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


# Dictionnaire contenant les abbréviations du dataset pour les instruments
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

# Liste contenant uniquement les abbréviations pour la construction des données
INSTRUMENTS = list(INSTRUMENTS_FULL_NAME.keys())
TEST_LENGTH = 500

base_path = '../data/'
test_data_path = base_path + 'test_data.csv'
train_data_path = base_path + 'train_data.csv'



def spectro_extract(file):
    """
    Define function that takes in a file path and returns features in an array

    Input : 
        - file : string, file path to the sound wo want to retrieve the spectrogram from

    Output : 
        - spectrogram : np.darray[], the spectrogram of the sound
    """
    # Get wave representation : y : waveform, s : sampling rate
    y, sr = librosa.load(file)
    n_mels = 128
    
    #get the mel-scaled spectrogram
    spectrogram = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=n_mels,fmax=11250)
    spectrogram = librosa.power_to_db(spectrogram, ref=np.max)

    # Normalise the spectrogram by bands of 4 lines
    # We also tried MinMaxScaler, by band and with the whole spectrogram
    for i in range(0, n_mels, 4):
        scaler = StandardScaler()
        spectrogram[i:i+4] = scaler.fit_transform(spectrogram[i:i+4])

    # Data augmentation : add noise to the spectrograms (old)
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
    # return image.img_to_array(image_pil)

    return spectrogram

#train_df = pd.read_csv(train_data_path)
#train_track_names = pd.unique(train_df['Track Name'])
#oui = spectro_extract('../data/' + train_track_names[0])
#print(oui.shape)

#Take tracks as an input and output multiple tracks
def create_test_tracks(file_name):
    """
    Create multiple tracks from a single file. Each track spans 3 seconds of audio to correspond to the input size of the model.
    
    Input : 
     - filename : string, path to file with extension
     
    Output : 
    - total_spectres : list(np.darray[128, 130]), list of spectrograms.
    """
    y, sr = librosa.load(file_name)
    spectre = spectro_extract(file_name)
    #Get length on y axis
    length = spectre.shape[1]
    target_size = 130
    number_of_subtracks = length // target_size
    total_spectres = [spectre[:, i*target_size:(i+1)*target_size] for i in range(number_of_subtracks)]

    return total_spectres

#Create labels
def one_hot_encode(y):
    """Creates the one-hot encoding for the instruments
    
    Input : 
        - y : list, the list of instruments in a track

    Output : 
        - y : list, the one-hot encoding of the instruments
    """
    for i in range(len(y)):
        y[i] = [1 if instr in y[i] else 0 for instr in INSTRUMENTS]
    return y

def create_model():
    """
    Creates and returns a model for the project. It takes in a spectrogram and outputs the instruments in the track.

    Output : 
        - model : Sequential, a model that determines the instruments in a track
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

# Model inspired by https://github.com/jeffprosise/Deep-Learning/blob/master/Audio%20Classification%20(CNN).ipynb
# We tested a bunch on this model but failed to use it correctly / was too big for our machines
def create_optimal_model():
    """
    Creates and returns a model for the project. It takes in a spectrogram and outputs the instruments in the track.

    Output : 
        - model : Sequential, a model that determines the instruments in a track
    """
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
            # We tested 256 for this Dense layer, without any success
            keras.layers.Dense(1024, activation='relu'),
            keras.layers.Dense(len(INSTRUMENTS_FULL_NAME), activation='sigmoid')
        ]
    )

    optimizer = keras.optimizers.Adam(learning_rate=0.00001)
    model.compile(loss='categorical_crossentropy', optimizer=optimizer, metrics=['accuracy'])
    return model

# Display grid search results (mean, std)
def display_cv_results(search_results):
    print('Best score = {:.4f} using {}'.format(search_results.best_score_, search_results.best_params_))
    means = search_results.cv_results_['mean_test_score']
    stds = search_results.cv_results_['std_test_score']
    params = search_results.cv_results_['params']
    for mean, stdev, param in zip(means, stds, params):
        print('mean test accuracy +/- std = {:.4f} +/- {:.4f} with: {}'.format(mean, stdev, param))  
        # Print keys of the cv_results_ dictionary
        print(search_results.cv_results_.keys())

# explicit function to normalize array : old function
# def normalize_2d(matrix):
#     norm = np.linalg.norm(matrix)
#     matrix = matrix/norm  # normalized matrix
#     return matrix


#Create x_train dataset
print("### TRAIN DATA ###")
train_df = pd.read_csv(train_data_path)
train_track_names = pd.unique(train_df['Track Name'])

x_train = [spectro_extract('../data/' + file) for file in tqdm(train_track_names)]

for i in tqdm(range(len(x_train))):
    x_train[i] = tf.convert_to_tensor(x_train[i], dtype=tf.float32)

#Get the labels of x_train
y_train = [train_df[train_df['Track Name'] == name]['Instrument'].values.tolist() for name in tqdm(train_track_names)]
y_train = one_hot_encode(y_train)

for i in tqdm(range(len(y_train))):
    y_train[i] = tf.convert_to_tensor(y_train[i], dtype=tf.float32)

x_train = np.array(x_train)
y_train = np.array(y_train)

#Save the lists
with open("x_train_norm", "wb") as fp:   #Pickling
    pickle.dump(x_train, fp)
with open("y_train_norm", "wb") as fp:   #Pickling
    pickle.dump(y_train, fp)

#Create x_test dataset
print("### TEST DATA ###")
test_df = pd.read_csv(test_data_path)
test_track_names = pd.unique(test_df['Track Name'])

x_test = [create_test_tracks('../data/' + file + '.wav') for file in tqdm(test_track_names)]
for i in tqdm(range(len(x_test))):
    x_test[i] = tf.convert_to_tensor(x_test[i], dtype=tf.float32)

#Get the labels of x_test
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

#Save the lists
with open("x_test_norm", "wb") as fp:   #Pickling
    pickle.dump(x_test, fp)
with open("y_test_norm", "wb") as fp:   #Pickling
    pickle.dump(y_test, fp)

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
# Code snippet that shows a spectrogram 
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


#Compute a grid search 3-fold cross validation
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


#We save the model only if the val loss decreases
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

#Get loss and accuracy of the test set
score = loaded.evaluate(x_test, y_test, verbose=0)
print("Test loss : ", score[0])
print("Test accuracy", score[1])
#Plot the results (graphs)
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