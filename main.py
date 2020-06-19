import librosa
from scipy.io import wavfile
import pathlib
import os
from matplotlib import pyplot as plt
import numpy as np
from utilities import filecontrol


def main():
    """path_to_read_american = pathlib.Path("data/American/wav")
    path_to_write_american = pathlib.Path("data/American/chunks")
    path_to_read_british = pathlib.Path("data/British/wav")
    path_to_write_british = pathlib.Path("data/British/chunks")

    filecontrol.get_chunks(path_to_read_american, path_to_write_american)
    filecontrol.get_chunks(path_to_read_british, path_to_write_british)"""

    filecontrol.get_spectrograms(pathlib.Path("data/American/chunks"), pathlib.Path("data/American/spectrograms"))


if __name__ == "__main__":
    main()
