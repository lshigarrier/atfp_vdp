import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from utils import load_yaml


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

    def __init__(self, param):
        """
        Load the data from the TSAGI_COMP file into self.data
        self.data is a dictionary -> key is time, value is a list of aircraft states
        aircraft state is a list: [idac, time, lon, lat, alt, speed, head, vz, cong]
        """
        data      = {}
        # list of lists containing the values of each parameter
        val_list  = [[] for _ in range(9)]
        # list of tuples containing the min and max of each parameter
        min_max   = []

        # Read the file
        with open(param['path']) as f:
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
                for i in range(9):
                    val_list[i].append(point[i])
                time = point[1]
                if time in data:
                    data[time].append(point)
                else:
                    data[time] = [point]
                line = f.readline()
        print('Reading done')

        self.times = list(data.keys())
        self.nb_times = len(self.times)

        # Compute the min and max of each parameter
        for i in range(9):
            min_max.append((min(val_list[i]), max(val_list[i])))
        self.min_max = min_max

        # Normalize all values and compute the maximum number of a/c at the same timestamp
        max_list = []
        for t in self.times:
            max_list.append(len(data[t]))
            for i in range(len(data[t])):
                for j in range(9):
                    data[t][i][j] = self.normalize(data[t][i][j], j)
            data[t] = torch.tensor(data[t])
        self.data   = data
        self.max_ac = np.amax(np.array(max_list))
        print('Normalization done')

        self.y_tensor = None
        self.x_tensor = None
        self.t_in     = int(param['T_in'])
        self.t_out    = int(param['T_out'])
        self.compute_output(param)
        # self.compute_input()
        print('Preprocessing done')

    def __len__(self):
        return self.nb_times - self.t_in - self.t_out + 1

    def __getitem__(self, item):
        input_seq = torch.zeros(self.max_ac*self.t_in, 8)
        start = 0
        for time in range(item, item+self.t_in):
            t = self.times[time]
            nb_ac = self.data[t].shape[0]
            input_seq[start:start+nb_ac, :] = self.data[t][:, :8]
            start += nb_ac
        return input_seq, self.y_tensor[item+self.t_in:item+self.t_in+self.t_out, :, :]


    def normalize(self, value, coord):
        """
        Normalize a value corresponding to a given coordinate
        coord = 0 -> aircraft ID
        coord = 1 -> time
        coord = 2 -> longitude
        coord = 3 -> latitude
        coord = 4 -> altitude
        coord = 5 -> ground speed
        coord = 6 -> heading
        coord = 7 -> vertical speed
        coord = 8 -> congestion metric
        """
        return (value - self.min_max[coord][0]) / (self.min_max[coord][1] - self.min_max[coord][0])

    def compute_output(self, param):
        """
        Compute the ground truth congestion map -> max congestion per cell
        If x << log_thr => log(1 + x/log_thr) ~= x/log_thr -> linear in x for small values
        If x >> log_thr => log(1 + x/log_thr) ~= log(x) - log(log_thr) -> log in x for high values
        If x = 0 => log(1 + x/log_thr) = 0
        Normalize the congestion between 0 and 1
        """
        nb_lon   = int(param['nb_lon'])
        nb_lat   = int(param['nb_lat'])
        log_thr  = float(param['log_thr'])
        y_tensor = torch.zeros(self.nb_times, nb_lon, nb_lat)
        max_val = np.log(1 + 1/log_thr)
        for time in range(self.nb_times):
            t = self.times[time]
            for i in range(len(self.data[t])):
                s_lon = int(self.data[t][i][2]*nb_lon)
                s_lat = int(self.data[t][i][3]*nb_lat)
                if s_lon > nb_lon - 1 or s_lat > nb_lat - 1 or s_lon < 0 or s_lat < 0:
                    continue
                val = cont2dis(np.log(1 + self.data[t][i][-1]/log_thr), max_val, param['nb_classes'])
                if val > y_tensor[time, s_lon, s_lat]:
                    y_tensor[time, s_lon, s_lat] = val
        self.y_tensor = y_tensor

    def compute_input(self):
        """
        Compute the inputs for the models
        The a/c states are organised by position, with zero padding
        """
        x_tensor = torch.zeros(self.nb_times, self.max_ac*6)
        for t in self.times:
            states = torch.tensor(sorted(self.data[t], key=lambda l:l[2]+10*l[3]))
            states = states[:,2:8].flatten()
            x_tensor[t, :states.shape[0]] = states[:]
        self.x_tensor = x_tensor


def main():
    param = load_yaml()
    dataset = TsagiSet(param)
    print(f'Nb of timestamps: {dataset.nb_times}')
    print(f'Dataset length: {len(dataset)}')
    print(f'Max nb of a/c: {dataset.max_ac}')
    loader = DataLoader(dataset,
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
    main()