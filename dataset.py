
import soundfile as sf
import librosa
import librosa.display

import numpy as np
import matplotlib.pyplot as plt
import torch
import torchvision
from PIL import Image
import os
import sys
from glob import glob
import constants
from tqdm import tqdm
import preprocessing

from torch.utils.data import Dataset


class MapsDataset(Dataset):
    def __init__(self,audio_path,tsv_path):

        self.audio_path = audio_path
        self.tsv_path = tsv_path

        self.data = []

        print("Loading Maps...("+audio_path+")")

        files = os.listdir(audio_path)

        for file in tqdm(files):
                (filename, ext) = os.path.splitext(file)
                if ext == ".flac":
                    self.load(filename)
               
            
        print("Maps loaded (" + str(len(files)) +") files")


    def load(self, filename):


        saved_data_path = self.audio_path+ filename+ ".pt"
        audio = self.audio_path + filename+ ".flac"
        tsv = self.tsv_path + filename+".tsv"

        if os.path.exists(saved_data_path) and (REBUILD == False):
            return torch.load(saved_data_path)


        audio,sr = sf.read(audio)
        assert(sr == constants.SAMPLE_RATE)

        spec = librosa.feature.melspectrogram(audio,sr, n_fft=1024, hop_length=512, n_mels=229)
      #  spec = librosa.power_to_db(spec, ref=np.max)



       # spec = np.abs(librosa.cqt(audio, sr=sr))
       # spec = librosa.amplitude_to_db(spec, ref=np.max)
       # librosa.display.specshow(librosa.amplitude_to_db(C, ref=np.max),sr=sr, x_axis='time', y_axis='cqt_note')
       # plt.colorbar(format='%+2.0f dB')
       # plt.title('Constant-Q power spectrum')
       # plt.tight_layout()
       # plt.show()


        '''
        librosa.display.specshow(spec, sr=sr, hop_length=512, x_axis='time', y_axis='mel');
        plt.colorbar(format='%+2.0f dB');
        plt.show()
        '''

        tsv = np.loadtxt(tsv,delimiter='\t',skiprows=1)

        for onset, offset, note, vel in tsv:

            relative_note = int(note) - constants.MIN_MIDI # we need to add + MIN_MIDI to network result to get midi pitch.
            sliced_spec = preprocessing.crop_spectrogram(spec,onset)
            self.data.append((sliced_spec,relative_note)) # we should save this ??

     
    def __getitem__(self, index):
        return self.data[index]


    def __len__(self):
        return len(self.data)




