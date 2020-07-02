import constants
import librosa
import torch
from model import Net
import soundfile as sf
import preprocessing

def to_midi(audio_file):
    
    net = Net() 
    net.load_state_dict(torch.load("model.pt"))
    net.eval()

    audio,sr = sf.read(audio_file)
    assert(sr == constants.SAMPLE_RATE)
    spec = librosa.feature.melspectrogram(audio,sr, n_fft=constants.FFT_SIZE, hop_length=constants.MEL_HOP_LENGTH, n_mels=constants.N_MEL)

    with torch.no_grad():
        onsets = preprocessing.detect_onsets(spec)



