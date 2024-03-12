import numpy as np
import librosa
import keras
import matplotlib.pyplot as plt

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

"""
_, _, s, _, _ = feature_extract("../data/Train/tru/[tru][cla]1870__1.wav")
print(s.shape)"""
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
model.save('audio_model.h5')

optimizer = keras.optimizers.Adam(learning_rate=0.001)
model.compile(loss="categorical_crossentropy", optimizer=optimizer, metrics=["accuracy"])

hist = model.fit(x_train, y_train, batch_size=64, epochs=5, validation_split=0.2)
score = model.evaluate(x_test, y_test, verbose=0)
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
""" EXTRAIRE DES FICHIERS WAV D'UNE CERTAINE LONGUEUR
from pydub import AudioSegment
t1 = t1 * 1000 #Works in milliseconds
t2 = t2 * 1000
newAudio = AudioSegment.from_wav("oldSong.wav")
newAudio = newAudio[t1:t2]
newAudio.export('newSong.wav', format="wav")"""