import pathlib
from utilities import filecontrol


def main():
    """
    # cut files into pieces
    path_to_read_american = pathlib.Path("data/American/wav")
    path_to_write_american = pathlib.Path("data/American/chunks")
    path_to_read_british = pathlib.Path("data/British/wav")
    path_to_write_british = pathlib.Path("data/British/chunks")

    filecontrol.get_chunks(path_to_read_american, path_to_write_american)
    filecontrol.get_chunks(path_to_read_british, path_to_write_british)
    """

    """
    # get final samples for neural netwroks as spectrograms
    filecontrol.get_spectrograms(pathlib.Path("data/American/chunks"), pathlib.Path("data/American/spectrograms"))
    filecontrol.get_spectrograms(pathlib.Path("data/British/chunks"), pathlib.Path("data/British/spectrograms"))
    """

    # extract paths and labels as lists. Label 0 is for American, 1 is for British
    filenames = filecontrol.get_absolute_paths("C:/Projects/Accent Classifier/data/American/spectrograms")
    labels = [0] * len(filenames)
    filenames += filecontrol.get_absolute_paths("C:/Projects/Accent Classifier/data/British/spectrograms")
    labels += [1] * (len(filenames) - len(labels))




if __name__ == "__main__":
    main()
