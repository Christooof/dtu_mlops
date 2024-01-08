import torch
import torch.nn as nn
import torchvision.transforms as transforms
from torch.optim import Adam
from torch.utils.data import DataLoader
from torchvision.datasets import MNIST
from torchvision.utils import save_image
from .__init__ import _PATH_DATA


mnist_transform = transforms.Compose([transforms.ToTensor()])
N_train=60000
N_test=5000
def test_data():
    dataset = MNIST(_PATH_DATA, transform=mnist_transform, train=True, download=True)
    assert len(dataset) == N_train #for training and N_test for test

    for data, label in dataset:
        assert data.shape in [(1, 28, 28), (784,)] 


     # Verify all labels are represented
    all_labels = set()
    for _, label in dataset:
        all_labels.add(label)
    assert len(all_labels) == 10  # Assuming there are 10 classes in MNIST dataset (digits 0-9)
    # You can modify this based on the number of classes in your dataset