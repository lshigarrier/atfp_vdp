import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from utils import load_yaml


def tsagi2frame(input_path='./data/20200718_C_NEW.TSAGI_COMP', output_path='./data/20200718_C_NEW.feather'):
    """
    Convert a TSAGI file into a feather file
    """
    print('Converting TSAGI to feather ...')
    data_list = []
    with open(input_path) as f:
        # The first line is the number of trajectories
        _ = f.readline()
        line = f.readline()
        while line != "":
            point = line.split()
            # 0->idac, 3->time, 4->lon, 5->lat, 6->alt, 7->speed, 8->head, 9->vz, 10->cong
            point = [float(point[0]),
                     float(point[3]),
                     float(point[4]),
                     float(point[5]),
                     float(point[6]),
                     float(point[7]),
                     float(point[8]),
                     float(point[9]),
                     float(point[10])]
            data_list.append(point)
            line = f.readline()
    data = pd.DataFrame(data_list, columns=['idac', 'time', 'lon', 'lat', 'alt', 'speed', 'head', 'vz', 'cong'])
    data.to_feather(output_path)
    print('... done!')


def cont2dis(value, max_val, nb_classes):
    """
    Transform a continuous value (regression) into a discrete value (classification)
    value must be in [0, max_val]
    nb_classes is the number of classes
    The classes are labeled as 1/nb_classes, 2/nb_classes, ..., (nb_classes-1)/nb_classes, 1
    """
    classes = np.linspace(0, max_val, num=nb_classes+1, endpoint=True)
    for i in range(1, len(classes)):
        if value <= classes[i]:
            return i/nb_classes
    raise RuntimeError


class TsagiSet(Dataset):

    def __init__(self, param, train=True):
        """
        Load the data from a feather file
        Normalize the data
        Create the ground truth
        """
        self.data = pd.read_feather(param['path'])
        print('Data loaded')

        # Compute the min and max of each column, then normalize all columns
        self.mins = self.data.min()
        self.maxs = self.data.max()
        self.data = (self.data - self.mins)/(self.maxs - self.mins)

        # Compute the maximum number of a/c at the same timestamp
        self.max_ac = self.data.groupby(['time'])['time'].count().max()

        # Initialize various attributes
        self.t_in          = int(param['T_in'])
        self.t_out         = int(param['T_out'])
        self.split_ratio   = param['split_ratio']
        self.times         = self.data['time'].tolist()
        self.total_seq     = len(self.times) - self.t_in - self.t_out + 1
        self.len_seq       = self.t_in - self.t_out
        self.sup_seq       = round(self.split_ratio * (self.total_seq + 4 * (1 - self.len_seq)) / 2 - 1)
        self.train         = train

        self.time_starts, self.timestamps = self.get_time_slices()
        self.output_tensor                = self.compute_output(param)
        print('Preprocessing done')

    def __len__(self):
        if self.train:
            return self.total_seq - 2*(2*self.len_seq + self.sup_seq - 1)
        else:
            return 2*(self.sup_seq + 1)

    def __getitem__(self, item):
        input_seq  = torch.zeros(self.t_in, self.max_ac, 8)
        time_start = self.time_starts[item]
        idx        = self.timestamps.index(time_start)
        for t in range(self.t_in):
            frame  = self.data.loc[self.data['time'] == self.timestamps[idx+t]]
            frame  = frame.drop(['time', 'cong'], axis=1)
            frame  = frame.sort_values(by=['idac'])
            frame  = frame.drop(['idac'])
            tensor = torch.tensor(frame.values)
            input_seq[t, :tensor.shape[0], :] = tensor[:]
        return input_seq, self.output_tensor[idx+self.t_in:idx+self.t_in+self.t_out, ...]

    def get_time_slices(self):
        train_seq = round((self.total_seq - 2*(self.len_seq + self.sup_seq))/3)
        if self.train:
            time_starts = self.times[:train_seq-self.len_seq+1]\
                        + self.times[train_seq+self.len_seq+self.sup_seq:2*train_seq+self.sup_seq+1]\
                        + self.times[2*train_seq+2*(self.len_seq+self.sup_seq):len(self.times)-self.len_seq+1]
            timestamps = self.times[:train_seq]\
                       + self.times[train_seq+self.len_seq+self.sup_seq:2*train_seq+self.len_seq+self.sup_seq]\
                       + self.times[2*(train_seq+self.len_seq+self.sup_seq):]
        else:
            time_starts = self.times[train_seq:train_seq+self.sup_seq+1] +\
                          self.times[2*train_seq+self.len_seq+self.sup_seq:2*train_seq+self.len_seq+2*self.sup_seq+1]
            timestamps = self.times[train_seq:train_seq+self.len_seq+self.sup_seq] +\
                         self.times[2*train_seq+self.len_seq+self.sup_seq:2*(train_seq+self.len_seq+self.sup_seq)]
        return time_starts, timestamps

    def compute_output(self, param):
        """
        Compute the ground truth congestion map -> max congestion per cell
        If x << log_thr => log(1 + x/log_thr) ~= x/log_thr -> linear in x for small values
        If x >> log_thr => log(1 + x/log_thr) ~= log(x) - log(log_thr) -> log in x for high values
        If x = 0 => log(1 + x/log_thr) = 0
        """
        nb_lon   = int(param['nb_lon'])
        nb_lat   = int(param['nb_lat'])
        log_thr  = float(param['log_thr'])
        output_tensor = torch.zeros(len(self.timestamps), nb_lon, nb_lat)
        max_val = np.log(1 + 1/log_thr)
        for t, time in enumerate(self.timestamps):
            for i_lon in range(nb_lon):
                for i_lat in range(nb_lat):
                    c_lon = i_lon/nb_lon
                    c_lat = i_lat/nb_lat
                    frame = self.data.loc[(self.data['time'] == time)
                            & (self.data['lon'] < c_lon+1/nb_lon/2) & (self.data['lon'] >= c_lon-1/nb_lon/2)
                            & (self.data['lat'] < c_lat+1/nb_lat/2) & (self.data['lat'] >= c_lat-1/nb_lat/2)]
                    if frame.empty:
                        continue
                    val   = frame['cong'].max()
                    output_tensor[t, i_lon, i_lat] = cont2dis(np.log(1 + val/log_thr), max_val, param['nb_classes'])
        return output_tensor


def main():
    param = load_yaml()
    trainset = TsagiSet(param, train=True)
    testset  = TsagiSet(param, train=False)
    print(f'Nb of timestamps: {len(trainset.times)}')
    print(f'Nb of sequences: {trainset.total_seq}')
    print(f'Trainset length: {len(trainset)}')
    print(f'Testset length: {len(testset)}')
    print(f'Max nb of a/c: {trainset.max_ac}')
    loader = DataLoader(testset,
                        batch_size=param['batch_size'],
                        shuffle=True,
                        pin_memory=True,
                        num_workers=1)
    for i, (x,y) in enumerate(loader):
        print(f'Batch: {i}')
        print(x.shape)
        print(torch.max(x).item())
        print(torch.min(x).item())
        print(y.shape)
        print(torch.max(y).item())
        print(torch.min(y).item())


if __name__ == '__main__':
    # tsagi2frame()
    main()
