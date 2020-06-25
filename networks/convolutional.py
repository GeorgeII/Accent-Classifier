import torch
from torch.utils.data import Dataset, DataLoader
from torchvision.models import resnet50
import cv2
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from tqdm import tqdm
import numpy as np


NETWORK_INPUT_SIZE = (224, 224)


class ConvolutionalNetwork():
    def __init__(self, device="cpu"):
        """
        Device can be "cuda" or "cpu". The latter is the default.
        """
        self.__model = resnet50(pretrained=True)
        self.__device = device
        self.__batch_size = 5

        for param in self.__model.parameters():
            param.requires_grad = False

        # Transfer learning itself. Replace the outer layer with 2 outputs only
        num_ftrs = self.__model.fc.in_features
        self.__model.fc = torch.nn.Linear(num_ftrs, 2)
        self.__model.to(self.__device)

    def train_model(self, filenames, labels, test_size=0.2, batch_size=5, epochs=50):
        """
        Train the neural network. This class is needed to encapsulate all the tricky and messy parts.
        :param test_size: proportion of test dataset to all files
        :param filenames: absolute paths to spectrograms
        :param labels: labels to each file
        :param batch_size: quantity of images in one batch
        """
        self.__batch_size = batch_size
        self.__epochs = epochs

        train_filenames, test_filenames, train_labels, test_labels = train_test_split(filenames, labels,
                                                                                      test_size=test_size, shuffle=True)

        train_dataset = SpectrogramDataset(train_filenames, train_labels)
        train_dataloader = DataLoader(train_dataset, shuffle=True, batch_size=self.__batch_size)
        test_dataset = SpectrogramDataset(test_filenames, test_labels)
        test_dataloader = DataLoader(test_dataset, shuffle=True, batch_size=self.__batch_size)

        criterion = torch.nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(self.__model.parameters(), lr=0.0005)

        for epoch in tqdm(range(epochs)):
            for i, (inputs, labels) in enumerate(train_dataloader):
                inputs = inputs.to(torch.device(self.__device))
                labels = labels.to(torch.device(self.__device))

                optimizer.zero_grad()

                outputs = self.__model(inputs)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()

            self.__run_test_on_epoch(epoch, test_dataloader)
        self.__model.eval()

    def __run_test_on_epoch(self, epoch, test_loader):
        self.__model.eval()
        with torch.no_grad():
            test_accuracy = []
            test_real = []
            for batch_x, batch_y in tqdm(test_loader):
                outputs = self.__model(batch_x.to(self.__device)).detach().cpu().numpy()
                test_accuracy.append(outputs)
                test_real.append(batch_y.detach().cpu().numpy())
            print("\nEpoch", epoch + 1, "test accuracy",
                  accuracy_score(np.hstack(test_real), np.argmax(np.vstack(test_accuracy), axis=1)))
        self.__model.train()

    def predict(self, filenames):
        """
        Predicts each class probability and return the average result
        :return np.ndarray with [m, 2] shape, where m is the length of filenames parameter
        """
        test_dataset = SpectrogramDataset(filenames, np.zeros(len(filenames)))
        test_dataloader = DataLoader(test_dataset, batch_size=self.__batch_size)

        outputs = []
        with torch.no_grad():
            for batch_x, batch_y in test_dataloader:
                prediction = self.__model(batch_x.to(self.__device)).detach().cpu().numpy()
                outputs.append(prediction)

        return np.vstack(outputs)

    def save_weights(self, path):
        torch.save(self.__model.state_dict(), path)

    def load_weights(self, path, device="cpu"):
        self.__model.load_state_dict(torch.load(path), map_location=torch.device(device))
        self.__model.eval()

    def deep_save(self, path):
        """
        Saves the whole model, i.e. paths, weights, criterions, optimizers, etc.
        """
        torch.save(self.__model, path)

    def deep_load(self, path, device="cpu"):
        """
        Loads a deep saved model.
        """
        self.__model = torch.load(path, map_location=torch.device(device))
        self.__model.eval()

    def get_model(self):
        return self.__model

    def set_model(self, trained_model):
        self.__model = trained_model


class SpectrogramDataset(Dataset):
    def __init__(self, absolute_filenames, labels):
        """
        IMPORTANT: ABSOLUTE paths to files is required
        :param absolute_filenames:
        :param labels:
        """
        self.__filenames = absolute_filenames
        self.__labels = labels

    def __len__(self):
        return len(self.__filenames)

    def __getitem__(self, item):
        filename = self.__filenames[item]
        label = self.__labels[item]

        image = cv2.imread(str(filename))
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = resize(image, NETWORK_INPUT_SIZE)
        image = add_pad(image, NETWORK_INPUT_SIZE)

        image = torch.tensor(image, dtype=torch.float).permute(2, 0, 1) / 255.

        return image, label


def resize(img, shape):
    scale = min(shape[0] * 1.0 / img.shape[0], shape[1] * 1.0 / img.shape[1])
    if scale != 1:
        img = cv2.resize(img, dsize=None, fx=scale, fy=scale, interpolation=cv2.INTER_LINEAR)

    return img


def add_pad(img, shape):
    """
    Adds padding to make the image square
    """
    color_pick = img[0][0]
    padded_img = color_pick * np.ones(shape + img.shape[2:3], dtype=np.uint8)
    x_offset = int((padded_img.shape[0] - img.shape[0]) / 2)
    y_offset = int((padded_img.shape[1] - img.shape[1]) / 2)
    padded_img[x_offset:x_offset + img.shape[0], y_offset:y_offset + img.shape[1]] = img

    return padded_img
