#implement.py

import os
import sys
import glob
import numpy
import librosa
import librosa.feature
import tensorflow as tf
from keras.src.ops import threshold
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense
from tqdm import tqdm
import yaml
import json
N_MELS = 64
FRAMES = 5
N_FFT = 1024
HOP_LENGTH = 512
POWER = 2.0
INPUT_DIM = N_MELS * FRAMES
def file_load(wav_name, mono=False):
    """
    load .wav file.

    wav_name : str
        target .wav file
    sampling_rate : int
        audio file sampling_rate
    mono : boolean
        When load a multi channels file and this param True, the returned data will be merged for mono data

    return : numpy.array( float )
    """
    try:
        return librosa.load(wav_name, sr=None, mono=mono)
    except Exception as e:
        print(f'Error loading {e}')



def demux_wav(wav_name, channel=0):
    """
    demux .wav file.

    wav_name : str
        target .wav file
    channel : int
        target channel number

    return : numpy.array( float )
        demuxed mono data

    Enabled to read multiple sampling rates.

    Enabled even one channel.
    """
    try:
        multi_channel_data, sr = file_load(wav_name)
        if multi_channel_data.ndim <= 1:
            return sr, multi_channel_data

        return sr, numpy.array(multi_channel_data)[channel, :]

    except ValueError as msg:
        print(f'{msg}')


def file_to_vector_array(file_name,
                         n_mels=64,
                         frames=5,
                         n_fft=1024,
                         hop_length=512,
                         power=2.0):
    """
    convert file_name to a vector array.

    file_name : str
        target .wav file

    return : numpy.array( numpy.array( float ) )
        vector array
        * dataset.shape = (dataset_size, fearture_vector_length)
    """
    # 01 calculate the number of dimensions
    dims = n_mels * frames

    # 02 generate melspectrogram using librosa (**kwargs == param["librosa"])
    sr, y = demux_wav(file_name)
    mel_spectrogram = librosa.feature.melspectrogram(y=y,
                                                     sr=sr,
                                                     n_fft=n_fft,
                                                     hop_length=hop_length,
                                                     n_mels=n_mels,
                                                     power=power)

    # 03 convert melspectrogram to log mel energy
    log_mel_spectrogram = 20.0 / power * numpy.log10(mel_spectrogram + sys.float_info.epsilon)

    # 04 calculate total vector size
    vectorarray_size = len(log_mel_spectrogram[0, :]) - frames + 1

    # 05 skip too short clips
    if vectorarray_size < 1:
        return numpy.empty((0, dims), float)

    # 06 generate feature vectors by concatenating multi_frames
    vectorarray = numpy.zeros((vectorarray_size, dims), float)
    for t in range(frames):
        vectorarray[:, n_mels * t: n_mels * (t + 1)] = log_mel_spectrogram[:, t: t + vectorarray_size].T

    return vectorarray



def create_autoencoder_model(input_dim=INPUT_DIM):
    """Defines the autoencoder architecture."""
    input_layer = Input(shape=(input_dim,))
    h = Dense(64, activation="relu")(input_layer)
    h = Dense(64, activation="relu")(h)
    h = Dense(8, activation="relu")(h)  # Bottleneck layer
    h = Dense(64, activation="relu")(h)
    h = Dense(64, activation="relu")(h)
    output_layer = Dense(input_dim, activation=None)(h)
    return Model(inputs=input_layer, outputs=output_layer)

def getResult():

if __name__ == "__main__":
    with open("../result/result.yaml") as stream:
        param = yaml.safe_load(stream)
    # NORMAL_TRAIN_DIR = './dataset/normal_training'
    # NORMAL_VALIDATION_DIR = 'dataset/normal_validation'  # Used to set the threshold
    MODEL_WEIGHTS_PATH = '../model/model_fan_id_00_-6_dB_fan.weights.h5'

    threshold = param["fan_id_00_-6_dB_fan"]["threshold"]

    if not os.path.exists(MODEL_WEIGHTS_PATH):
        print("Error: Model files not found. Please run train_autoencoder.py first.")
        sys.exit(1)

        # Load the anomaly threshold
    # with open(MODEL_CONFIG_PATH, 'r') as f:
    #     config = json.load(f)
    #     threshold = config['threshold']

    model = create_autoencoder_model()
    model.load_weights(MODEL_WEIGHTS_PATH)
    files = ['E:\mimii_baseline\dataset\min6db\-6_dB_fan\\fan\id_00\\abnormal\\00000023.wav',
             "E:\mimii_baseline\dataset\min6db\-6_dB_fan\\fan\id_00\\abnormal\\00000037.wav",
             "E:\mimii_baseline\dataset\min6db\-6_dB_fan\\fan\id_00\\normal\\00000252.wav"]
    for file in files:
        audio_file = file
        # Example usage with a test WAV file
        test_files =file_to_vector_array(audio_file)
        print(test_files.size)

        if test_files.size == 0:
            print("Could not extract features. The file might be too short or corrupted.")
            sys.exit(1)

        reconstructed_audio = model.predict(test_files)
        mse = numpy.mean(numpy.power(test_files - reconstructed_audio, 2), axis=1)
        anomaly_score = numpy.percentile(mse, 99)

        print(f"Anomaly Score: {anomaly_score:.6f}")
        print(f"Threshold:     {threshold:.6f}")

        if anomaly_score > threshold:
            print("\nResult: ABNORMAL")
        else:
            print("\nResult: Normal")

