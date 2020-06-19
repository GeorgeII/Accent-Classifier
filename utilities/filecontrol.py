import librosa
from librosa import display
import os
import subprocess
import pathlib
import numpy as np
from matplotlib import pyplot as plt
from utilities import trimmer


def get_chunks(read_directory, write_directory):
    """
    Takes every WAV file in a given directory and cuts it into pieces which are called chunks.
    If there is a file named 'Speech of Bob' then it will be cut and saved as 'Speech of Bob0', 'Speech of Bob1', etc. in
    a given directory to write.
    Also removes pauses and quiet sounds.
    :param read_directory: pathlib.Path format preferably
    :param write_directory: pathlib.Path format preferably
    """
    for file in os.scandir(read_directory):
        if file.name.endswith(".wav"):
            y, sampling_rate = librosa.load(file)
            y = trimmer.trim_start_end(y, sampling_rate=sampling_rate)
            chunks = trimmer.trim_pauses_and_split_into_chunks(y, top_db=30, chunk_duration=4,
                                                               sampling_rate=sampling_rate)

            for idx, chunk in enumerate(chunks):
                filename = pathlib.Path(file).stem
                path_to_write = write_directory / pathlib.Path(filename + f'{idx}.wav')
                librosa.output.write_wav(path_to_write, chunk, sampling_rate)


def remove_all_wav(path):
    """
    Removes all the WAV files in a given directory.
    """
    for file in os.scandir(path):
        if file.name.endswith(".wav"):
            os.unlink(file.path)


def mp3_to_wav(read_directory, write_directory):
    """
    Converts MP3 file into WAV file. FFmpeg is required to be installed.
    Size of WAV files is much larger than MP3 so directory sizes increase tremendously.
    Each MP3 is converted into a PCM-encoded WAV file with 44,100 Hz sample rate and 16 bits per sample.
    """
    files_to_read = list(pathlib.Path(read_directory).glob('*.mp3'))

    for file in files_to_read:
        filename = pathlib.Path(file).stem
        file_to_write = write_directory / pathlib.Path(filename).with_suffix(".wav")
        subprocess.call(['ffmpeg', '-i', f'{file}', f'{file_to_write}'])


def get_spectrograms(read_directory, write_directory):
    """
    Transform given sound into a spectrogram and save it as an image.
    :param read_directory: pathlib.Path format preferably
    :param write_directory: pathlib.Path format preferably
    """
    for file in os.scandir(read_directory):
        if file.name.endswith(".wav"):
            y, sampling_rate = librosa.load(file)
            spectrogram = librosa.amplitude_to_db(np.abs(librosa.stft(y)), ref=np.max)
            display.specshow(spectrogram)

            filename = pathlib.Path(file).stem
            path_to_write = write_directory / pathlib.Path(filename).with_suffix(".png")
            plt.savefig(path_to_write)

#%%
y, sampling_rate = librosa.load(pathlib.Path("data/American/chunks/Inspiring Interview of Will Smith on December 2016 - How To Face Fear228.wav"))
spectrogram = librosa.amplitude_to_db(np.abs(librosa.stft(y)), ref=np.max)
display.specshow(spectrogram)
plt.show()
#%%
from scipy import signal

f, t, Zxx = signal.stft(y, sampling_rate, nperseg=512)
plt.pcolormesh(t, f, np.abs(Zxx), cmap='jet')
plt.show()