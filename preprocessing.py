import librosa.display
import matplotlib.pyplot as plt
import numpy
import sys
import constants

def crop_spectrogram(spectrogram,onset):

    index = (constants.SAMPLE_RATE * onset) / constants.MEL_HOP_LENGTH
    index = int(index)

    result = spectrogram[0:229,index:int(index+constants.BLOCK_SIZE_BIN)]
   
    librosa.display.specshow(result, sr=constants.SAMPLE_RATE, hop_length=512, x_axis='time', y_axis='mel');
    plt.colorbar(format='%+2.0f dB')
    plt.show()
  
    return result



def detect_onsets(spectrogram):
    # todo
    pass