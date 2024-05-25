import pandas as pd
import numpy as np


class CustomDataSet:
    def __init__(self,
                 sequence: [tuple],
                 cloatting: np.array,
                 protalitic: np.array,
                 CP: np.array):
        self.sequence = sequence
        self.cloatting = cloatting
        self.protalitic = protalitic
        self.CP = CP
        self.embedding = None

    def add_embedding(self,
                       embedding):

        self.embedding = embedding

    def get_embedding(self):
        return self.embedding


    # def __len__(self):
    #     return int(self.profile.size(dim=0))
    #
    #
    # def __getitem__(self, idx: int) -> tuple[Union[FloatTensor, str], Union[FloatTensor, str]]:
    #     return self.profile[idx, :], self.group[idx], self.name[idx]
    #
    #
    # def subset(self, indices: list) -> tuple[np.array, pd.Series, pd.Series]:
    #     profiles = self.profile[indices, :]
    #     groups = self.group.iloc[indices].reset_index(drop=True)
    #     names = self.name.iloc[indices].reset_index(drop=True)
    #     return profiles, groups, names