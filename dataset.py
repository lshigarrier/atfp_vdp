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

    def __init__(self, param, train=True, state_dim=6):
        """
        Load the data from a feather file
        Normalize the data
        Create the ground truth
        """
        self.train = train
        if self.train:
            print('Preprocessing training set')
        else:
            print('Preprocessing test set')

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
        self.times         = self.data['time'].drop_duplicates().sort_values().tolist()
        self.total_seq     = len(self.times) - self.t_in - self.t_out + 1
        self.len_seq       = self.t_in + self.t_out
        self.sup_seq       = round(self.split_ratio * (self.total_seq + 4 * (1 - self.len_seq)) / 2 - 1)
        self.state_dim     = state_dim

        self.time_starts, self.timestamps = self.get_time_slices()
        self.output_tensor                = self.compute_output(param)
        print('Preprocessing done')

    def __len__(self):
        if self.train:
            return self.total_seq - 2*(2*self.len_seq + self.sup_seq - 1)
        else:
            return 2*(self.sup_seq + 1)

    def __getitem__(self, item):
        input_seq = torch.zeros(self.t_in, self.max_ac, 6)
        time_start = self.time_starts[item]
        idx = self.timestamps.index(time_start)
        for t in range(self.t_in):
            frame = self.data.loc[self.data['time'] == self.timestamps[idx + t]]
            frame = frame.drop(['time', 'cong', 'int_lon', 'int_lat'], axis=1)
            frame = frame.sort_values(by=['idac'])
            frame = frame.drop(['idac'], axis=1)
            tensor = torch.tensor(frame.values)
            input_seq[t, :tensor.shape[0], :] = tensor[:]
        return input_seq, self.output_tensor[idx + self.t_in:idx + self.t_in + self.t_out, ...]

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
        max_val  = np.log(1 + 1/log_thr)

        self.data['int_lon'] = (self.data['lon']*(nb_lon - 1)).round()
        self.data['int_lat'] = (self.data['lat']*(nb_lon - 1)).round()

        frame = self.data.loc[:, ['time', 'int_lon', 'int_lat', 'cong']]
        frame = frame[frame['time'].isin(self.timestamps)]
        frame = frame.sort_values(by=['cong'], ascending=False).drop_duplicates(['time', 'int_lon', 'int_lat'])
        frame = frame.sort_values(by=['time', 'int_lon', 'int_lat'])

        frame['cong'] = (np.log(1 + frame['cong']/log_thr)/max_val*(param['nb_classes'] - 1)).round()
        frame['cong'] = frame['cong']/(param['nb_classes'] - 1)

        tensor = torch.tensor(frame.values)

        output_tensor = torch.zeros(len(self.timestamps), nb_lon, nb_lat)
        for row in tensor:
            t = self.timestamps.index(row[0].item())
            output_tensor[t, round(row[1].item()), round(row[2].item())] = row[3].item()
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
