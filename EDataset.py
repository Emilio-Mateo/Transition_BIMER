
import numpy as np
from torch.utils.data import Dataset
from sklearn.preprocessing import MinMaxScaler

class Transition(Dataset):
    def __init__(self, sz = 100, path0 = None, path1 = None, transform = None):
        data = np.load(path0)
        if path1:
            self.label = np.load(path1)
            self.Tlabel = True
        else:
            self.Tlabel = None
        xs, ts = data.shape
        self.sz = ts
        Xdata = np.zeros((ts,64,64))
        for i in range(ts):
                Xdata[i] = data[:,i].reshape(64,64)
                Xdata[i] -= np.min(Xdata[i])
                Xdata[i] /= np.max(Xdata[i])
        self.data = Xdata.astype(np.float32)
        self.transform = transform

    def __getitem__(self, index):

        if self.transform:
            sample_transformed = self.transform(self.data[index])
            if self.Tlabel:
                sample_labeled = (sample_transformed,int(self.label[index]))
                sample = sample_labeled
            else:
                sample = sample_transformed
        return sample
    
    def __len__(self):
        return self.sz


class Labels(Dataset):
    def __init__(self, sz = 100, path = None, transform = None):
        # data = np.load(path)[:50,:]
        data = path
        # xs, ts = data.shape
        self.sz = len(data)
        self.data = data.astype(np.float32)
        self.transform = transform

    def __getitem__(self, index):

        if self.transform:
            sample_transformed = torch.tensor(self.data[index])
        return sample_transformed
    
    def __len__(self):
        return self.sz
    
class LinearData(Dataset):
    def __init__(self, sz = 100, path0 = None, path1 = None, transform = None, 
                 validation = None, pathT = None):
        scaler = MinMaxScaler()
        if validation:
            train = np.load(pathT)
            scaler.fit(train)
        else:
            train = np.load(path0)
            scaler.fit(train)
    
        data = scaler.transform(np.load(path0))
        label = np.load(path1)
        sz, ts = data.shape
        self.sz = sz
        self.data = data.astype(np.float32)
        self.label = label.astype(np.float32)
        self.transform = transform

    def __getitem__(self, index):

        if self.transform:
            sample_transformed = torch.from_numpy(self.data[index])
            sample_labeled = (sample_transformed, int(self.label[index]))
        return sample_labeled
    
    def __len__(self):
        return self.sz