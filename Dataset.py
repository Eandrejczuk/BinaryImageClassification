import numpy as np
from torch.utils.data import Dataset,DataLoader
import sys
np.set_printoptions(threshold=sys.maxsize)

class TrainingDatasetClass(Dataset):

    def __init__(self, data, labels, transform=None):
        self.data = data
        self.labels = labels
        self.transform = transform

    def __getitem__(self, index):
        image = self.data[index]
        if self.transform:
            image = (image * 255).astype('uint8')
            image = self.transform(image)
        return (image, self.labels[index])

    def __len__(self):
        return len(self.labels)
