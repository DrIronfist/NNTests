from pathlib import Path
import pickle
import gzip
from matplotlib import pyplot
import torch
import neuralNetwork


FILENAME = "mnist.pkl.gz"
DATA_PATH = Path("data")
PATH = DATA_PATH / "mnist"
with gzip.open((PATH / FILENAME).as_posix(), "rb") as f:
        ((x_train, y_train), (x_valid, y_valid), _) = pickle.load(f, encoding="latin-1")

pyplot.imshow(x_train[0].reshape((28, 28)), cmap="gray")
print(x_train.shape)
x_train, y_train, x_valid, y_valid = map(
    torch.tensor, (x_train, y_train, x_valid, y_valid)
)
n, c = x_train.shape
bs = 64
xb = x_train[0:bs]
preds = neuralNetwork.model(xb)  # predictions
preds[0], preds.shape
print(preds[0], preds.shape)
