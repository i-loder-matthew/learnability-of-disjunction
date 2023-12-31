import os
import torch, torchtext
import pandas as pd
import numpy as np
from torch.utils.data import Dataset, DataLoader
from sklearn import preprocessing

from collections import Counter

class DisjunctionLanguageDataset(Dataset):
    """Disjunction artificial language dataset."""

    def __init__(self, path_to_file, transform = None) -> None:
        super().__init__()

        self.df = pd.read_pickle(path_to_file)
        self.transform = transform
        self.vocab = list(Counter([x for xs in self.df for x in xs]).keys())


    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.to_list()

        sample = self.df.iloc[idx]

        if self.transform:
            sample = self.transform(sample)


        return sample



class OneHotOutputs(object): # TODO clean this up; put all in terms of shape of trg

    def __init__(self, max_length, vocab = ['<sos>', 'a', '%', '<eos>', 'b', '', '*', '?', '&']):
        self.max_length = max_length
        self.vocab = vocab

    def __call__(self, sample):
        src = np.stack(sample.src)
        trg = sample.trg

        vocab_arr = [[x] for x in self.vocab]

        ohe = preprocessing.OneHotEncoder(handle_unknown="ignore")
        ohe.fit(vocab_arr)
        
        l_voc = len(self.vocab)

        if len(np.shape(sample)) == 1:

            out = np.zeros((1, self.max_length, l_voc))
            s_trans = [[x] for x in sample[1]]
            s_trans = ohe.transform(s_trans).A 

            for j,r in enumerate(s_trans):
                out[0, j, :] = r
        else:
            assert (len(np.shape(sample)) > 1)

            l_data = len(trg)
            out = np.zeros((l_data, self.max_length, l_voc))

            for i, s in enumerate(trg):
                s_trans = [[x] for x in s]
                s_trans = ohe.transform(s_trans).A 
                
                for j, r in enumerate(s_trans):
                    out[i, j, :] = r


        return {'src': torch.tensor(src, dtype=torch.float), 'trg': torch.tensor(np.array(out), dtype=torch.float)}


class ToTensor(object):
    
    def __init__(self):
        pass

    def __call__(self, sample):

        return {'src': torch.tensor(sample['src']), 'trg': torch.tensor(sample['trg'])}
