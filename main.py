import pathlib
import glob
import os
import numpy as np
from utilities import filecontrol
from utilities import trimmer
from networks.convolutional import ConvolutionalNetwork


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

    """
    # For training. Extract paths and labels as lists. Label 0 is for American, 1 is for British
    filenames = filecontrol.get_absolute_paths("/mnt/e89ecbe3-8e48-d601-e01e-41e38e48d601/datasets/Accent-Classifier/data/American/spectrograms")
    labels = [0] * len(filenames)
    filenames += filecontrol.get_absolute_paths("/mnt/e89ecbe3-8e48-d601-e01e-41e38e48d601/datasets/Accent-Classifier/data/British/spectrograms")
    labels += [1] * (len(filenames) - len(labels))
    print(len(labels), len(filenames))
    """



    '''
    # Test stage. Evaluates accuracy.
    cur_path = pathlib.Path(__file__).parent
    print(cur_path)

    filecontrol.mp3_to_wav(cur_path / pathlib.Path("data/test/American/mp3"), cur_path / pathlib.Path("data/test/American/wav"))
    filecontrol.mp3_to_wav(cur_path / pathlib.Path("data/test/British/mp3"), cur_path / pathlib.Path("data/test/British/wav"))

    print("Cutting WAV into chunks...")
    filecontrol.get_chunks(cur_path / pathlib.Path("data/test/American/wav"), cur_path / pathlib.Path("data/test/American/chunks"))
    filecontrol.get_chunks(cur_path / pathlib.Path("data/test/British/wav"), cur_path / pathlib.Path("data/test/British/chunks"))

    print("Getting spectrograms...")
    filecontrol.get_spectrograms(cur_path / pathlib.Path("data/test/American/chunks"), cur_path / pathlib.Path("data/test/American/spectrograms"))
    filecontrol.get_spectrograms(cur_path / pathlib.Path("data/test/British/chunks"), cur_path / pathlib.Path("data/test/British/spectrograms"))

    model = ConvolutionalNetwork()
    print(pathlib.Path(__file__).parent)
    model.deep_load(cur_path / pathlib.Path("models/deep_copy_convolutional.pt"))
    print(model.get_model())

    filenames = filecontrol.get_absolute_paths("/home/george/Projects/test-env/Accent-Classifier/data/test/American/spectrograms")
    labels = [0] * len(filenames)
    filenames += filecontrol.get_absolute_paths("/home/george/Projects/test-env/Accent-Classifier/data/test/British/spectrograms")
    labels += [1] * (len(filenames) - len(labels))
    print(len(filenames), len(labels))

    preds = np.argmax(model.predict(filenames), axis=1)
    print("Spectrograms independently: ", accuracy_score(labels, preds))

    print(np.sum(preds) / len(preds))
    '''



    # Inference stage. That's how it works in production
    cur_path = pathlib.Path(__file__).parent

    print("Converting mp3 to WAV...")
    filecontrol.mp3_to_wav(cur_path / pathlib.Path("data/inference/mp3"), cur_path / pathlib.Path("data/inference/wav"))

    print("Cutting WAV into chunks...")
    filecontrol.get_chunks(cur_path / pathlib.Path("data/inference/wav"), cur_path / pathlib.Path("data/inference/chunks"))

    print("Getting spectrograms...")
    filecontrol.get_spectrograms(cur_path / pathlib.Path("data/inference/chunks"), cur_path / pathlib.Path("data/inference/spectrograms"))


    model = ConvolutionalNetwork()
    model.deep_load(cur_path / pathlib.Path("models/deep_copy_convolutional.pt"))
    #print(model.get_model())
    
    filenames = filecontrol.get_absolute_paths("/home/george/Projects/test-env/Accent-Classifier/data/inference/spectrograms")
    print("Splitted into ", len(filenames), "parts")
    
    preds = np.argmax(model.predict(filenames), axis=1)
    print("Independent predicions: ", preds)
    print("Final prediction: ", np.sum(preds) / len(preds))
    
    accent = ""
    if np.rint(np.sum(preds)) == 0:
        accent = "American"
    else:
        accent = "British"
    
    print("It is " + accent + " English.")


    inference_dirs = ["wav", "chunks", "spectrograms"]
    for directory in inference_dirs:
        files = glob.glob(f'data/inference/{directory}/*')
        for f in files:
            os.remove(f)


if __name__ == "__main__":
    main()
