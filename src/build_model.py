import numpy as np
import librosa

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
    spectrogram = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=128,fmax=8000)  
    #temporally average spectrogram
    spectrogram = np.mean(spectrogram, axis = 1)
    
    #compute chroma energy
    chroma = librosa.feature.chroma_cens(y=y, sr=sr)
    #temporally average chroma
    chroma = np.mean(chroma, axis = 1)
    
    #compute spectral contrast
    contrast = librosa.feature.spectral_contrast(y=y, sr=sr)
    contrast = np.mean(contrast, axis= 1)
    
    return [harmonic, mfcc, spectrogram, chroma, contrast]