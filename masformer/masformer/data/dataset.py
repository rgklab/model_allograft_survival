import numpy as np
import pandas as pd
import torch
from torch import Tensor
from torch.utils.data import Dataset
from operator import itemgetter


class SRTR(Dataset):

    def __init__(
        self, 
        features, labels,
        train: bool = True) -> None:
        super().__init__()

        self.train = train
        self.max_time = features.shape[1]
        self.data = []

        temp = []
        for feature, label in zip(features, labels):
            feature = torch.from_numpy(feature).float()
            time, duration, is_observed = label[0], label[1], label[2]
            temp.append([time, duration, is_observed, feature])
        sorted_temp = sorted(temp, key=itemgetter(0))

        new_temp = sorted_temp if self.train else temp

        for time, duration, is_observed, feature in new_temp:
            duration = int(duration)
            num_pad = self.max_time - (duration + 1)
            mask = [1.]*(duration+1) + [0.]*num_pad
            if is_observed:
                label = duration * [1.] + (self.max_time - duration) * [0.]
                self.data.append([feature, torch.tensor(time).float(), 
                    torch.tensor(mask).float(), torch.tensor(label), torch.tensor(is_observed).bool()])
            else:
                label = self.max_time * [1.]
                self.data.append([feature, torch.tensor(time).float(), 
                    torch.tensor(mask).float(), torch.tensor(label), torch.tensor(is_observed).bool()])


    def __getitem__(self, idx: int):
        """Returns training/validation data for one patient.
        """
        if self.train:
            if idx == len(self.data) - 1:
                index_b = np.random.randint(len(self.data))
            else:
                # NOTE self.data is sorted
                is_observed_list = [x[4] for x in self.data]
                index_list = np.arange(0, len(self.data))

                where = np.where((torch.tensor(is_observed_list) == 1.) & (index_list > idx+1), index_list, np.nan)
                where = where[~np.isnan(where)]

                if where.size == 0:
                    index_b = np.random.randint(len(self.data))
                else:
                    index_b = int(np.random.choice(where))
            return [ [self.data[idx][i], self.data[index_b][i]] for i in range(len(self.data[idx])) ]
        else:

            # return self.data[idx]
            if idx == len(self.data) - 1:
                index_b = np.random.randint(len(self.data))
            else:
                # NOTE self.data is sorted
                is_observed_list = [x[4] for x in self.data]
                index_list = np.arange(0, len(self.data))

                where = np.where((torch.tensor(is_observed_list) == 1.) & (index_list > idx+1), index_list, np.nan)
                where = where[~np.isnan(where)]

                if where.size == 0:
                    index_b = np.random.randint(len(self.data))
                else:
                    index_b = int(np.random.choice(where))
            return [ [self.data[idx][i], self.data[index_b][i]] for i in range(len(self.data[idx])) ]



    def __len__(self) -> int:
        """Return the size of this dataset."""
        return len(self.data)