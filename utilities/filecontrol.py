import librosa
import os
import subprocess
import pathlib
import numpy as np
from matplotlib import pyplot as plt
from utilities import trimmer


def get_chunks(read_directory, write_directory, start_trim=15, end_trim=15, top_db=30):
    """
    Takes every WAV file from a given directory and cuts it into pieces which are called chunks.
    For instance, if there is a file named 'Speech of Bob' then it will be cut and saved as 'Speech of Bob0', 'Speech of Bob1', etc. in
    a given directory to write.
    Also removes pauses and quiet sounds.
    :param read_directory: pathlib.Path format preferably
    :param write_directory: pathlib.Path format preferably
    """
    for file in os.scandir(read_directory):
        if file.name.endswith(".wav"):
            y, sampling_rate = librosa.load(file)
            y = trimmer.trim_start_end(y, start_trim=start_trim, end_trim=end_trim, sampling_rate=sampling_rate)
            chunks = trimmer.trim_pauses_and_split_into_chunks(y, top_db=top_db, chunk_duration=4,
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
    Transforms all WAV sounds in a given directory into a spectrogram and saves them as images.

    Details: reads chunks of speech. Creates an image as 2x2 of combined images.
    In 1 row 1 column there's a simple STFT spectrogram. 1 row 2 column is a logarithmic scaled of STFT.
    In 2 row 1 column it's a logarithmic scale of squared STFT. 2 row 2 column is filter-banks applied to STFT.
    :param read_directory: pathlib.Path format preferably
    :param write_directory: pathlib.Path format preferably
    """
    for file in os.scandir(read_directory):
        if file.name.endswith(".wav"):
            y, sampling_rate = librosa.load(file)

            # STFT
            matrix_stft = np.abs(librosa.stft(y, n_fft=512))
            # STFT + filterbank
            melspectrogram = librosa.feature.melspectrogram(y=y, sr=sampling_rate)

            fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, gridspec_kw={'hspace': 0, 'wspace': 0})

            ax1.pcolormesh(np.abs(matrix_stft), cmap='jet')
            ax2.pcolormesh(np.log(np.abs(matrix_stft)), cmap='jet')
            ax3.pcolormesh(librosa.power_to_db(np.abs(matrix_stft) ** 2), cmap='jet')
            ax4.pcolormesh(librosa.power_to_db(melspectrogram), cmap='jet')

            ax1.axis('off')
            ax2.axis('off')
            ax3.axis('off')
            ax4.axis('off')
            plt.axis('off')

            filename = pathlib.Path(file).stem
            path_to_write = write_directory / pathlib.Path(filename).with_suffix(".png")
            plt.savefig(path_to_write)
            plt.close(fig)


def get_absolute_paths(directory):
    """
    Gives absolute paths of files inside of a given directory.
    :param directory: absolute path of directory with files
    :return: absolute paths of files inside the given directory
    """
    path = pathlib.Path(directory).glob('**/*')
    files = [x for x in path if x.is_file()]

    return files
