from librosa import effects
import numpy as np


def trim_start_end(audio, start_trim=10, end_trim=10, sampling_rate=22050):
    """
    Cuts some audio from start and end because usually it's music with an intro/outro or noise.

    :param audio: numpy array representing an audio signal
    :param start_trim: time in seconds that will be removed from the beginning
    :param end_trim: time in seconds that will be removed from the end
    :param sampling_rate: number of samples in every second
    :return: np.ndarray [shape=(n,) or (2, n)]
    """

    return audio[sampling_rate * start_trim : -sampling_rate * end_trim]


def trim_pauses_and_split_into_chunks(audio, top_db=30, chunk_duration=4, sampling_rate=22050):
    """
    Trims all the pauses in the file and cut it into chunks of a given sample_duration.
    For example, if a 50 seconds file is given then it will remove silence first (40 seconds remaining)
    and then will return 10 arrays with each representing a 4 second sample.
    All chunks are cut fully coherently so their stacking gives the trimmed file.
    :param audio: numpy array representing an audio signal
    :param top_db: The threshold (in decibels) below reference to consider as silence
    :param chunk_duration: cut into chunks of 'sample_duration' length
    :param sampling_rate: number of discrete samples in every second, not be confused with sample size we cut the audio into
    :return: list of np.ndarray [shape=(m, chunk_duration*sampling rate)]. m = ceil(len(audio) / chunk_duration)
    """
    intervals = effects.split(audio, top_db=top_db)

    # trim pauses
    trimmed_audio = np.empty(0)
    for i in range(len(intervals)):
        trimmed_audio = np.concatenate((trimmed_audio, audio[intervals[i][0]: intervals[i][1]]))

    # split into little chunks
    chunks = []
    step = chunk_duration * sampling_rate
    for i in range(0, len(trimmed_audio), step):
        chunks.append(audio[i:i + step])

    return chunks
