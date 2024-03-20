import numpy as np
import librosa
import keras
import matplotlib.pyplot as plt
import pandas as pd
from tqdm import tqdm
import pickle
#Skip Tensorflow warnings
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'


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
TEST_LENGTH = 20

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
    
    #get wave representation : y : waveform, s : sampling rate
    y, sr = librosa.load(file)
    
    #get the mel-scaled spectrogram
    spectrogram = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=128,fmax=45000)
    
    # Specshow the spectrogram
    # plt.figure(figsize=(10, 4))
    # librosa.display.specshow(librosa.power_to_db(spectrogram, ref=np.max), y_axis='mel', fmax=45000, x_axis='time')
    # plt.colorbar(format='%+2.0f dB')
    # plt.title('Mel spectrogram')
    # plt.tight_layout()
    # plt.show()
    
    return spectrogram

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
        keras.layers.Dense(len(INSTRUMENTS_FULL_NAME), activation='softmax')
    ]
)

model.summary()
optimizer = keras.optimizers.Adam(learning_rate=0.001)
model.compile(loss="categorical_crossentropy", optimizer=optimizer, metrics=["accuracy"])

#LOAD THE DATA
train_df = pd.read_csv(train_data_path)
train_track_names = pd.unique(train_df['Track Name'])
x_train = [spectro_extract('../data/' + file) for file in tqdm(train_track_names)]
with open("x_train", "wb") as fp:   #Pickling
    pickle.dump(x_train, fp)
y_train = [train_df[train_df['Track Name'] == name]['Instrument'].values.tolist() for name in tqdm(train_track_names)]
y_train = one_hot_encode(y_train)
with open("y_train", "wb") as fp:   #Pickling
    pickle.dump(y_train, fp)


test_df = pd.read_csv(test_data_path)
test_track_names = pd.unique(test_df['Track Name'])
x_test = [create_test_tracks('../data/' + file + '.wav') for file in tqdm(test_track_names)]
with open("x_test", "wb") as fp:   #Pickling
    pickle.dump(x_test, fp)
y_test = [test_df[test_df['Track Name'] == name]['Instrument'].values.tolist() for name in tqdm(test_track_names)]
y_test = one_hot_encode(y_test)
with open("y_test", "wb") as fp:   #Pickling
    pickle.dump(y_test, fp)

with open("x_train", "rb") as fp:   # Unpickling
    x_train = pickle.load(fp)
with open("y_train", "rb") as fp:   # Unpickling
    y_train = pickle.load(fp)
with open("x_test", "rb") as fp:   # Unpickling
    x_test = pickle.load(fp)
with open("y_test", "rb") as fp:   # Unpickling
    y_test = pickle.load(fp)




""" EXTRAIRE DES FICHIERS WAV D'UNE CERTAINE LONGUEUR
from pydub import AudioSegment
t1 = t1 * 1000 #Works in milliseconds
t2 = t2 * 1000
newAudio = AudioSegment.from_wav("oldSong.wav")
newAudio = newAudio[t1:t2]
newAudio.export('newSong.wav', format="wav")"""

'''
checkpoint_path = "audio_model.h5"
model_checkpoint_callback = callbacks.ModelCheckpoint(
    filepath=checkpoint_path,
    monitor='val_loss',
    mode='min',
    save_best_only=True,
    verbose=1
)


hist = model.fit(x_train, y_train, batch_size=64, epochs=5, validation_split=0.2, callbacks=[model_checkpoint_callback])

loaded = load_model("model_checkpoint_callback")

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
'''