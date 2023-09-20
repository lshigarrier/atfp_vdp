import pandas as pd
import matplotlib.pyplot as plt
import torch
from torch.utils.data import Dataset
# from torch.utils.data import DataLoader


def tsagi2frame(input_path='./data/20200718_C.TSAGI_COMP1', output_path='./data/20200718_C_CONV.feather'):
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


class TsagiSet(Dataset):

    def __init__(self, param, train=True, quantiles=None):
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

        # Load data
        self.data = pd.read_feather(param['path'])

        # Filter data
        self.data, self.nb_trajs = self.filter_data()

        # Compute the min and max of each column, then normalize all columns
        self.mins = self.data.min()
        self.maxs = self.data.max()
        self.data = (self.data - self.mins)/(self.maxs - self.mins)

        # Compute the maximum number of a/c at the same timestamp
        self.max_ac = self.data.groupby(['time'])['time'].count().max()

        # Quantiles of the congestion values to define the classes
        if quantiles is None:
            self.quantiles = []
        else:
            self.quantiles = quantiles

        # Initialize various attributes
        self.nb_lon       = int(param['nb_lon'])
        self.nb_lat       = int(param['nb_lat'])
        self.nb_alt       = int(param['nb_alt'])
        self.t_in         = int(param['T_in'])
        self.t_out        = int(param['T_out'])
        self.split_ratio  = param['split_ratio']
        self.times        = self.data['time'].drop_duplicates().sort_values().tolist()
        self.total_seq    = len(self.times) - self.t_in - self.t_out + 1
        self.len_seq      = self.t_in + self.t_out
        self.sup_seq      = round(self.split_ratio * (self.total_seq + 4 * (1 - self.len_seq)) / 2 - 1)
        self.state_dim    = param['state_dim']
        self.predict_spot = param['predict_spot']
        if self.predict_spot:
            self.spot = param['spot']

        self.data['idx_lon'] = (self.data['lon']*(self.nb_lon - 1)).round()
        self.data['idx_lat'] = (self.data['lat']*(self.nb_lat - 1)).round()
        self.data['idx_alt'] = (self.data['alt']*(self.nb_alt - 1)).round()
        self.time_starts, self.timestamps = self.get_time_slices()
        print('Building outputs')
        self.output_tensor = self.compute_output(param['nb_classes'])
        self.output_mask   = self.compute_mask()
        self.output_tensor = self.output_tensor.masked_fill(self.output_mask.eq(0), -1)
        print('Preprocessing done')

    def __len__(self):
        if self.train:
            return self.total_seq - 2*(2*self.len_seq + self.sup_seq - 1)
        else:
            return 2*(self.sup_seq + 1)

    def __getitem__(self, item):
        input_seq = torch.zeros(self.t_in, self.max_ac*self.state_dim)
        time_start = self.time_starts[item]
        idx = self.timestamps.index(time_start)
        for t in range(self.t_in):
            frame = self.data.loc[self.data['time'] == self.timestamps[idx + t]]
            frame = frame.loc[:, ['idac', 'lon', 'lat', 'alt', 'speed', 'head', 'vz']]
            frame = frame.sort_values(by=['idac'])
            frame = frame.drop(['idac'], axis=1)
            tensor = torch.tensor(frame.values).transpose(0, 1)
            input_seq[t, :tensor.numel()] = tensor.flatten()
        if self.predict_spot:
            out_seq = -2*torch.ones(self.t_out+1, 1, dtype=torch.long)
        else:
            out_seq = -2*torch.ones(self.t_out+1, self.nb_lon*self.nb_lat, dtype=torch.long)
        out_seq[1:, :] = self.output_tensor[idx + self.t_in:idx + self.t_in + self.t_out, :]
        return input_seq, out_seq

    def filter_data(self):
        """
        Remove trajectories whose max is below FL200
        Remove points that are below FL100
        :return: self.data
        """
        max_level = self.data.loc[self.data.groupby('idac')['alt'].idxmax()].reset_index(drop=True)
        max_level = max_level.loc[max_level['alt'] > 20000]
        max_level = max_level.loc[:, 'idac'].tolist()
        frame = self.data.loc[self.data['idac'].isin(max_level)]
        frame = frame.loc[frame['alt'] > 10000]
        return frame, len(max_level)

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

    def get_class(self, value, nb_classes):
        for i in range(1, nb_classes-1):
            if value < self.quantiles[i-1]:
                return i
        return nb_classes - 1

    def compute_output(self, nb_classes):
        """
        Compute the ground truth congestion map -> max congestion per cell
        """
        if self.predict_spot:
            frame = self.data.loc[:, ['time', 'idx_lon', 'idx_lat', 'idx_alt', 'cong']]
            frame = frame.loc[(self.data['idx_lon']==self.spot[0]) &
                              (self.data['idx_lat']==self.spot[1]) &
                              (self.data['idx_alt']==self.spot[2])]
            frame = frame.drop(['idx_lon', 'idx_lat', 'idx_alt'], axis=1)
            frame = frame[frame['time'].isin(self.timestamps)]
            frame = frame.sort_values(by=['cong'], ascending=False).drop_duplicates(['time'])

            if len(self.quantiles) == 0:
                for i in range(1, nb_classes-1):
                    self.quantiles.append(frame['cong'].quantile(q=i/(nb_classes-1)))

            tensor = torch.tensor(frame.values)
            output_tensor = torch.zeros(len(self.timestamps), 1, dtype=torch.long)
            for row in tensor:
                t = self.timestamps.index(row[0].item())
                output_tensor[t, 0] = self.get_class(row[1].item(), nb_classes)
            return output_tensor

        else:
            frame = self.data.loc[:, ['time', 'idx_lon', 'idx_lat', 'cong', 'alt']]
            frame = frame.loc[(frame['alt'] > (29000 - self.mins['alt'])/(self.maxs['alt'] - self.mins['alt'])) &
                              (frame['alt'] < (36000 - self.mins['alt'])/(self.maxs['alt'] - self.mins['alt']))]
            frame.drop('alt', axis=1)
            frame = frame[frame['time'].isin(self.timestamps)]
            frame = frame.sort_values(by=['cong'], ascending=False).drop_duplicates(['time', 'idx_lon', 'idx_lat'])

            if len(self.quantiles) == 0:
                for i in range(1, nb_classes-1):
                    self.quantiles.append(frame['cong'].quantile(q=i/(nb_classes-1)))

            tensor = torch.tensor(frame.values)
            output_tensor = torch.zeros(len(self.timestamps), self.nb_lon, self.nb_lat, dtype=torch.long)
            for row in tensor:
                t = self.timestamps.index(row[0].item())
                output_tensor[t, round(row[1].item()), round(row[2].item())] = self.get_class(row[3].item(), nb_classes)
            return output_tensor.view(len(self.timestamps), -1)

    def compute_mask(self):
        output_sum  = self.output_tensor.sum(dim=0)
        return output_sum.masked_fill(output_sum.bool(), 1)

    def get_hot_spots(self):
        """
        Compute the sum of complexity in each cell
        """
        frame = self.data.loc[:, ['time', 'idx_lon', 'idx_lat', 'idx_alt', 'cong']]
        frame = frame.sort_values(by=['cong'], ascending=False).drop_duplicates(['time', 'idx_lon', 'idx_lat', 'idx_alt'])
        frame = frame.groupby(['idx_lon', 'idx_lat', 'idx_alt'])['cong'].sum()
        frame = frame.sort_values(ascending=False)
        print(frame.head(10))
        return frame


def dataset_balance(param, tsagi):
    total = tsagi.output_tensor.ne(-1).int().sum().item()
    weights = []
    for i in range(param['nb_classes']):
        count = torch.eq(tsagi.output_tensor, i).float().sum().item()
        print(f'Class {i}: {int(count)} ; {100.*count/total:.2f}%')
        weights.append(total/count)
    min_w = min(weights)
    for i in range(param['nb_classes']):
        print(f'Weight {i}: {weights[i]/min_w:.2f}')


def dataset_iterate(loader):
    for i, (x,y) in enumerate(loader):
        print(f'Batch: {i}')
        print(x.shape)
        print(x.max().item())
        print(x.min().item())
        print(y.shape)
        print(y.max().item())
        print(y.min().item())


def main():
    from utils import load_yaml
    param = load_yaml()
    for key in param:
        print(f'{key}: {param[key]}')
    trainset = TsagiSet(param, train=True)
    testset  = TsagiSet(param, train=False, quantiles=trainset.quantiles)
    testset.output_mask = trainset.output_mask[:]
    print(f'Mins\n{trainset.mins}\nMaxs\n{trainset.maxs}')
    print(f'Nb of trajs: {trainset.nb_trajs}')
    print(f'Max nb of a/c: {trainset.max_ac}')
    print(f'Nb of rows: {trainset.data.shape[0]}')
    print(f'Nb of timestamps: {len(trainset.times)}')
    print(f'Nb of sequences: {trainset.total_seq}')
    print(f'Trainset length: {len(trainset)}')
    print(f'Trainset quantiles: {trainset.quantiles}')
    print('Trainset balance')
    dataset_balance(param, trainset)
    print(f'Testset length: {len(testset)}')
    print('Testset balance')
    dataset_balance(param, testset)
    print('Hot-spots')
    trainset.get_hot_spots()
    # loader = DataLoader(testset, batch_size=param['batch_size'], shuffle=True, pin_memory=True, num_workers=1)
    # dataset_iterate(loader)
    from plots import plot_one_img
    _ = plot_one_img(trainset.output_mask.view(param['nb_lon'], param['nb_lat']))
    plt.show()


if __name__ == '__main__':
    # tsagi2frame()
    main()
