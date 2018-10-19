import torch
import os
import numpy as np
import scipy.io
import torch.utils.data as Data

# Data set size
train_num = 70000
val_num = 30000

# Supported modulations
mods = ['BPSK', 'DQPSK', 'GFSK', 'GMSK', 'OQPSK', 'PAM4', 'PAM8', 'PSK8', 'QAM16', 'QAM64', 'QPSK']
class_num = len(mods)


class ModulationDataset(Data.Dataset):
    """Define dataset to load RF signal data"""

    def __init__(self, is_train):
        self.is_train = is_train

        if self.is_train:
            file_name = 'train.pt'
        else:
            file_name = 'val.pt'

        loc_name = file_name
        if not os.path.exists(loc_name):
            raise RuntimeError('Do not find file: %s' % loc_name)
        else:
            data_label = torch.load(loc_name)
            self.input = data_label['input']
            self.label = data_label['label']

    def __getitem__(self, index):
        return self.input[index], self.label[index]

    def __len__(self):
        if self.is_train:
            return train_num * class_num
        else:
            return val_num * class_num


def separate_train_val(mods, train_num, val_num):

    data = scipy.io.loadmat('C:/matlab_sim/batch100000_symbols128_sps8_baud1_snr45.dat')
    train_input = []
    train_label = []
    val_input = []
    val_label = []
    for mod in mods:
        real = torch.tensor(np.array(data[mod].real)).unsqueeze_(1)
        imag = torch.tensor(np.array(data[mod].imag)).unsqueeze_(1)
        signal = torch.cat([real, imag], dim=1)
        train_input.append(signal[:train_num])
        val_input.append(signal[-val_num:])
        train_label.append(mods.index(mod) * torch.ones(train_num))
        val_label.append(mods.index(mod) * torch.ones(val_num))

    train_input = torch.cat(train_input, dim=0)
    train_label = torch.cat(train_label, dim=0)
    val_input = torch.cat(val_input, dim=0)
    val_label = torch.cat(val_label, dim=0)

    train = {'input': train_input, 'label': train_label}
    val = {'input': val_input, 'label': val_label}

    torch.save(train, 'train.pt')
    torch.save(val, 'val.pt')


if __name__ == '__main__':
    separate_train_val(mods, train_num, val_num)
