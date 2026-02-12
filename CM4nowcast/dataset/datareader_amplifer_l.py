import pandas as pd
import os
import numpy as np
import h5py
import torch
from torch.utils.data import Dataset
from torch.utils.data import DataLoader

os.environ["HDF5_USE_FILE_LOCKING"] = 'FALSE'
TYPES = ['vis', 'ir069', 'ir107', 'vil', 'lght']
DEFAULT_CATALOG = '/root/autodl-tmp/SEVIR/CATALOG_VIL_LEFT_2.csv'
DEFAULT_DATA_HOME = '/root/autodl-tmp/SEVIR'
FRAME_TIMES = np.arange(-120.0, 120.0, 5) * 60

# TODO : MANY!

# Example function of how to create pytorch dataloader with sevir dataset
def get_sevir_loader_event(batch_size=3,
                     x_img_types=['vil'],
                     y_img_types=None,
                     catalog=DEFAULT_CATALOG,
                     start_date=None,
                     end_date=None,
                     datetime_filter=None,
                     catalog_filter=None,
                     unwrap_time=False,
                     sevir_data_home=DEFAULT_DATA_HOME,
                     shuffle=False,
                     pin_memory=True,
                     output_type=np.float32,
                     normalize_x=None,
                     normalize_y=None,
                     num_workers=0,
                     show_time = False,
                     data_amplifer = 1
                     ):
    dataset = SEVIR(x_img_types,
                    y_img_types,
                    catalog,
                    start_date,
                    end_date,
                    datetime_filter,
                    catalog_filter,
                    unwrap_time,
                    sevir_data_home,
                    output_type,
                    normalize_x,
                    normalize_y,
                    show_time = show_time,
                    data_amplifer = 1
                    )
    return DataLoader(dataset,
                      batch_size=batch_size,
                      shuffle=shuffle,
                      pin_memory=pin_memory,
                      num_workers=num_workers)


def get_sevir_loader(batch_size=3,
                     x_img_types=['vil'],
                     y_img_types=None,
                     catalog=DEFAULT_CATALOG,
                     start_date=None,
                     end_date=None,
                     datetime_filter=None,
                     catalog_filter=None,
                     unwrap_time=False,
                     sevir_data_home=DEFAULT_DATA_HOME,
                     shuffle=False,
                     pin_memory=True,
                     output_type=np.float32,
                     normalize_x=None,
                     normalize_y=None,
                     num_workers=0,
                     show_time = False,
                     data_amplifer = 1
                     ):
    dataset = NowcastGenerator1(x_img_types,
                    y_img_types,
                    catalog,
                    start_date,
                    end_date,
                    datetime_filter,
                    catalog_filter,
                    unwrap_time,
                    sevir_data_home,
                    output_type,
                    normalize_x,
                    normalize_y,
                    show_time = show_time,
                    data_amplifer = data_amplifer
                    )
    return DataLoader(dataset,
                      batch_size=batch_size,
                      shuffle=shuffle,
                      pin_memory=pin_memory,
                      num_workers=num_workers)

def get_sevir_loader_spilt_l(batch_size=3,
                     x_img_types=['vil'],
                     y_img_types=None,
                     catalog=DEFAULT_CATALOG,
                     start_date=None,
                     end_date=None,
                     datetime_filter=None,
                     catalog_filter=None,
                     unwrap_time=False,
                     sevir_data_home=DEFAULT_DATA_HOME,
                     shuffle=False,
                     pin_memory=True,
                     output_type=np.float32,
                     normalize_x=None,
                     normalize_y=None,
                     num_workers=0,
                     show_time = False,
                     data_spilt = 0.8,
                     data_amplifer = 1,
                     random_seed = 42
                     ):
    dataset = NowcastGenerator1(x_img_types,
                    y_img_types,
                    catalog,
                    start_date,
                    end_date,
                    datetime_filter,
                    catalog_filter,
                    unwrap_time,
                    sevir_data_home,
                    output_type,
                    normalize_x,
                    normalize_y,
                    show_time = show_time,
                    data_amplifer = data_amplifer
                    )
    train_size = int(data_spilt * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset_split, val_dataset_split = torch.utils.data.random_split(dataset, [train_size, val_size],generator=torch.Generator().manual_seed(random_seed))
    return DataLoader(train_dataset_split,batch_size=batch_size,shuffle=shuffle,pin_memory=pin_memory,num_workers=num_workers, drop_last=True), DataLoader(val_dataset_split,batch_size=batch_size,shuffle=shuffle,pin_memory=pin_memory,num_workers=num_workers, drop_last=True)





class SEVIR(Dataset):
    """
    Sequence class for generating batches from SEVIR

    Parameters
    ----------
    catalog  str or pd.DataFrame
        name of SEVIR catalog file to be read in, or an already read in and processed catalog
    x_img_types  list
        List of image types to be used as model inputs.  For types, run SEVIRSequence.get_types()
    y_img_types  list or None
       List of image types to be used as model targets (if None, __getitem__ returns only x_img_types )
    sevir_data_home  str
       Directory path to SEVIR data
    catalog  str
       Name of SEVIR catalog CSV file.
    start_date   datetime
       Start time of SEVIR samples to generate
    end_date    datetime
       End time of SEVIR samples to generate
    datetime_filter   function
       Mask function applied to time_utc column of catalog (return true to keep the row).
       Pass function of the form   lambda t : COND(t)
       Example:  lambda t: np.logical_and(t.dt.hour>=13,t.dt.hour<=21)  # Generate only day-time events
    catalog_filter  function
       Mask function applied to entire catalog dataframe (return true to keep row).
       Pass function of the form lambda catalog:  COND(catalog)
       Example:  lambda c:  [s[0]=='S' for s in c.id]   # Generate only the 'S' events
    unwrap_time   bool
       If True, single images are returned instead of image sequences
    output_type  np.dtype
       dtype of generated tensors
    normalize_x  list of tuple
       list the same size as x_img_types containing tuples (scale,offset) used to
       normalize data via   X  -->  (X-offset)*scale.  If None, no scaling is done
    normalize_y  list of tuple
       list the same size as y_img_types containing tuples (scale,offset) used to
       normalize data via   X  -->  (X-offset)*scale

    Returns
    -------
    SEVIRSequence generator

    Examples
    --------

        # Get just Radar image sequences
        vil_seq = SEVIRSequence(x_img_types=['vil'],batch_size=16)
        X = vil_seq.__getitem__(1234)  # returns list the same size as x_img_types passed to constructor

        # Get ir satellite+lightning as X,  radar for Y
        vil_ir_lght_seq = SEVIRSequence(x_img_types=['ir107','lght'],y_img_types=['vil'],batch_size=4)
        X,Y = vil_ir_lght_seq.__getitem__(420)  # X,Y are lists same length as x_img_types and y_img_types

        # Get single images of VIL
        vil_imgs = SEVIRSequence(x_img_types=['vil'], batch_size=256, unwrap_time=True, shuffle=True)

        # Filter out some times
        vis_seq = SEVIRSequence(x_img_types=['vis'],batch_size=32,unwrap_time=True,
                                start_date=datetime.datetime(2018,1,1),
                                end_date=datetime.datetime(2019,1,1),
                                datetime_filter=lambda t: np.logical_and(t.dt.hour>=13,t.dt.hour<=21))

    """

    def __init__(self,
                 x_img_types=['vil'],
                 y_img_types=None,
                 catalog=DEFAULT_CATALOG,
                 start_date=None,
                 end_date=None,
                 datetime_filter=None,
                 catalog_filter=None,
                 unwrap_time=False,
                 sevir_data_home=DEFAULT_DATA_HOME,
                 output_type=np.float32,
                 normalize_x=None,
                 normalize_y=None,
                 show_time = False,
                 data_amplifer = 1
                 ):
        self._samples = None
        self._hdf_files = {}
        self.x_img_types = x_img_types
        self.y_img_types = y_img_types
        if isinstance(catalog, (str,)):
            self.catalog = pd.read_csv(catalog, parse_dates=['time_utc'], low_memory=False)
        else:
            self.catalog = catalog

        self.datetime_filter = datetime_filter
        self.catalog_filter = catalog_filter
        self.start_date = start_date
        self.end_date = end_date
        self.unwrap_time = unwrap_time
        self.sevir_data_home = sevir_data_home
        self.output_type = output_type
        self.normalize_x = normalize_x
        self.normalize_y = normalize_y
        self.show_time = show_time
        self.data_amplifer = data_amplifer
        if normalize_x:
            assert (len(normalize_x) == len(x_img_types))
        if normalize_y:
            assert (len(normalize_y) == len(y_img_types))

        if self.start_date:
            self.catalog = self.catalog[self.catalog.time_utc > self.start_date]
        if self.end_date:
            self.catalog = self.catalog[self.catalog.time_utc <= self.end_date]
        if self.datetime_filter:
            self.catalog = self.catalog[self.datetime_filter(self.catalog.time_utc)]

        if self.catalog_filter:
            self.catalog = self.catalog[self.catalog_filter(self.catalog)]

        self._compute_samples()
        self._open_files()

    def _compute_samples(self):
        """
        Computes the list of samples in catalog to be used. This sets
           self._samples

        """
        # locate all events containing colocated x_img_types and y_img_types
        imgt = self.x_img_types
        if self.y_img_types:
            imgt = list(set(imgt + self.y_img_types))  # remove duplicates
        imgts = set(imgt)
        filtcat = self.catalog[np.logical_or.reduce([self.catalog.img_type == i for i in imgt])]
        # remove rows missing one or more requested img_types
        filtcat = filtcat.groupby('id').filter(lambda x: imgts.issubset(set(x['img_type'])))
        # If there are repeated IDs, remove them (this is a bug in SEVIR)
        filtcat = filtcat.groupby('id').filter(lambda x: x.shape[0] == len(imgt))
        self._samples = filtcat.groupby('id').apply(lambda df: df_to_series(df, imgt, unwrap_time=self.unwrap_time))

    def _open_files(self):
        """
        Opens HDF files
        """
        imgt = self.x_img_types
        if self.y_img_types:
            imgt = list(set(imgt + self.y_img_types))  # remove duplicates
        hdf_filenames = []
        for t in imgt:
            hdf_filenames += list(np.unique(self._samples[f'{t}_filename'].values))
        self._hdf_files = {}
        for f in hdf_filenames:
            print('Opening HDF5 file for reading', f)
            self._hdf_files[f] = h5py.File(self.sevir_data_home + '/' + f, 'r')

    def __getitem__(self, idx):
        """
        batch_x = self.x[idx * self.batch_size:(idx + 1) * self.batch_size]
        batch_y = self.y[idx * self.batch_size:(idx + 1) * self.batch_size]

        return np.array([
            resize(imread(file_name), (200, 200))
               for file_name in batch_x]), np.array(batch_y)
        """
        data = {}
        data = read_data(self._samples.iloc[idx], data, self._hdf_files, self.unwrap_time)
        X = [data[t].astype(self.output_type) for t in self.x_img_types]
        if self.normalize_x:
            X = [normalize(X[k], s[0], s[1]) for k, s in enumerate(self.normalize_x)]
        if self.y_img_types is not None:
            Y = [data[t].astype(self.output_type) for t in self.y_img_types]
            if self.normalize_y:
                Y = [normalize(Y[k], s[0], s[1]) for k, s in enumerate(self.normalize_y)]
            return X, Y
        else:
            if self.show_time:
                time_row = (self._samples.iloc[idx][0],self._samples.iloc[idx][1])
                return X, time_row
            else:
                return X

    def __len__(self):
        return (2*self.data_amplifer + 1)*len(self._samples)


def read_data(row, data, hdf_files, unwrap_time=False):
    """
    Reads data from data object
    :param row: series with fields IMGTYPE_filename IMGTYPE_index, IMGTYPE_time_index
    :param data: data object
    :param hdf_files: hdf_file handles to read from
    :param unwrap_time: boolean for unwrapping time field
    :return:
    """
    image_types = np.unique([x.split('_')[0] for x in list(row.keys())])
    for t in image_types:
        f_name = row[f'{t}_filename']
        idx = row[f'{t}_index']
        if unwrap_time:
            time_idx = row[f'{t}_time_index']
            t_slice = slice(time_idx, time_idx + 1)
        else:
            t_slice = slice(0, None)
        # Need to bin lght counts into grid
        if t == 'lght':
            lightning_data = hdf_files[f_name][idx][:]
            data_i = lightning_to_hist(lightning_data, t_slice)
        else:
            data_i = hdf_files[f_name][t][idx:idx + 1, :, :, t_slice]
        data[t] = np.concatenate((data[t], data_i), axis=0) if (t in data) else data_i
    return data

def lightning_to_hist(data, t_slice=slice(0, None)):
    """
    Converts Nx5 lightning data matrix into a XYT histogram
    :param data: lightning event data
    :param t_slice: temporal dimension
    :return: XYT histogram of lightning data
    """

    out_size = (48, 48, len(FRAME_TIMES)) if t_slice.stop is None else (48, 48, 1)
    if data.shape[0] == 0:
        return np.zeros((1,) + out_size, dtype=np.float32)

    # filter out points outside the grid
    x, y = data[:, 3], data[:, 4]
    m = np.logical_and.reduce([x >= 0, x < out_size[0], y >= 0, y < out_size[1]])
    data = data[m, :]
    if data.shape[0] == 0:
        return np.zeros((1,) + out_size, dtype=np.float32)

    # Filter/separate times
    t = data[:, 0]
    if t_slice.stop is not None:  # select only one time bin
        if t_slice.stop > 0:
            tm = np.logical_and(t >= FRAME_TIMES[t_slice.stop - 1],
                                t < FRAME_TIMES[t_slice.stop])
        else:  # special case:  frame 0 uses lightning from frame 1
            tm = np.logical_and(t >= FRAME_TIMES[0], t < FRAME_TIMES[1])
        data = data[tm, :]
        z = np.zeros(data.shape[0], dtype=np.int64)
    else:  # compute z coordinate based on bin location times
        z = np.digitize(t, FRAME_TIMES) - 1
        z[z == -1] = 0  # special case:  frame 0 uses lightning from frame 1

    x = data[:, 3].astype(np.int64)
    y = data[:, 4].astype(np.int64)

    k = np.ravel_multi_index(np.array([y, x, z]), out_size)
    n = np.bincount(k, minlength=np.prod(out_size))
    return np.reshape(n, out_size).astype(np.float32)[np.newaxis, :]


def normalize(x, scale, offset, reverse=False):
    """
    Normalize data or reverse normalization
    :param x: data array
    :param scale: const scaling value
    :param offset: const offset value
    :param reverse: boolean undo normalization
    :return: normalized x array
    """
    if reverse:
        return x / scale + offset
    else:
        return (x-offset) * scale


def df_to_series(df, image_types, n_frames=49, unwrap_time=False):
    """
    This looks like it takes a data frame and turns it into a sequence like data frame
    :param df: pandas data frame
    :param image_types: image types to extract
    :param n_frames: number of frames to extract
    :param unwrap_time: boolean for whether or not to set time index
    :return: sequence data frame
    """
    d = {}
    df = df.set_index('img_type')
    for i in image_types:
        s = df.loc[i]
        idx = s.file_index if i != 'lght' else s.id
        if unwrap_time:
            d.update({f'{i}_filename': [s.file_name] * n_frames,
                      f'{i}_index': [idx] * n_frames,
                      f'{i}_time_index': range(n_frames)})
        else:
            d.update({f'{i}_filename': [s.file_name],
                      f'{i}_index': [idx]})
    return pd.DataFrame(d)

class NowcastGenerator(SEVIR):
    """
    Generator that loads full VIL sequences, and spilts each
    event into three training samples, each 12 frames long.

    Event Frames:  [-----------------------------------------------]
                   [----13-----][---12----]
                               [----13----][----12----]
                                          [-----13----][----12----]
    """
    def __getitem__(self, idx):
        """
        """
        if self.show_time:
            X, time_row= super(NowcastGenerator, self).__getitem__(idx)  # N,L,W,49
            x1,x2,x3 = X[0][:,:,:,:13],X[0][:,:,:,12:25],X[0][:,:,:,24:37]
            y1,y2,y3 = X[0][:,:,:,13:25],X[0][:,:,:,25:37],X[0][:,:,:,37:49]
            Xnew = np.concatenate((x1,x2,x3),axis=0)
            Ynew = np.concatenate((y1,y2,y3),axis=0)
            return [Xnew],[Ynew],time_row
        else:
            X = super(NowcastGenerator, self).__getitem__(idx)  # N,L,W,49
            x1,x2,x3 = X[0][:,:,:,:13],X[0][:,:,:,12:25],X[0][:,:,:,24:37]
            y1,y2,y3 = X[0][:,:,:,13:25],X[0][:,:,:,25:37],X[0][:,:,:,37:49]
            Xnew = np.concatenate((x1,x2,x3),axis=0)
            Ynew = np.concatenate((y1,y2,y3),axis=0)
            return [Xnew],[Ynew]

class NowcastGenerator1(SEVIR):
    """
    Generator that loads full VIL sequences, and spilts each
    event into three training samples, each 12 frames long.

    Event Frames:  [-----------------------------------------------]
                   [----13-----][---12----]
                               [----13----][----12----]
                                          [-----13----][----12----]
    """
    def __getitem__(self, idx):
        """
        """
        if self.data_amplifer == 1:
            idx1 = idx // 3
            if self.show_time:
                if idx%3 == 0:
                    X, time_row = super(NowcastGenerator1, self).__getitem__(idx1)  # N,L,W,49
                    x1 = X[0][:,:,:,:13]
                    y1 = X[0][:,:,:,13:25]
                elif idx%3 == 1:
                    X, time_row = super(NowcastGenerator1, self).__getitem__(idx1)  # N,L,W,49
                    x1 = X[0][:,:,:,12:25]
                    y1 = X[0][:,:,:,25:37]
                elif idx%3 == 2:
                    X, time_row = super(NowcastGenerator1, self).__getitem__(idx1)  # N,L,W,49
                    x1 = X[0][:,:,:,24:37]
                    y1 = X[0][:,:,:,37:49]

                return torch.from_numpy(x1).permute(0,3,1,2)[0], torch.from_numpy(y1).permute(0,3,1,2)[0], time_row
            else:
                if idx%3 == 0:
                    X = super(NowcastGenerator1, self).__getitem__(idx1)  # N,L,W,49
                    x1 = X[0][:,:,:,:13]
                    y1 = X[0][:,:,:,13:25]
                elif idx%3 == 1:
                    X = super(NowcastGenerator1, self).__getitem__(idx1)
                    x1 = X[0][:,:,:,12:25]
                    y1 = X[0][:,:,:,25:37]
                elif idx%3 == 2:
                    X = super(NowcastGenerator1, self).__getitem__(idx1)
                    x1 = X[0][:,:,:,24:37]
                    y1 = X[0][:,:,:,37:49]

                return torch.from_numpy(x1).permute(0,3,1,2)[0], torch.from_numpy(y1).permute(0,3,1,2)[0]

        elif self.data_amplifer == 2:
            idx1 = idx // 5
            if self.show_time:
                if idx%5 == 0:
                    X, time_row = super(NowcastGenerator1, self).__getitem__(idx1)  # N,L,W,49
                    x1 = X[0][:,:,:,:13]
                    y1 = X[0][:,:,:,13:25]
                elif idx%5 == 1:
                    X, time_row = super(NowcastGenerator1, self).__getitem__(idx1)  # N,L,W,49
                    x1 = X[0][:,:,:,6:19]
                    y1 = X[0][:,:,:,19:31]
                elif idx%5 == 2:
                    X, time_row = super(NowcastGenerator1, self).__getitem__(idx1)  # N,L,W,49
                    x1 = X[0][:,:,:,12:25]
                    y1 = X[0][:,:,:,25:37]
                elif idx%5 == 3:
                    X, time_row = super(NowcastGenerator1, self).__getitem__(idx1)  # N,L,W,49
                    x1 = X[0][:,:,:,18:31]
                    y1 = X[0][:,:,:,31:43]
                elif idx%5 == 4:
                    X, time_row = super(NowcastGenerator1, self).__getitem__(idx1)  # N,L,W,49
                    x1 = X[0][:,:,:,24:37]
                    y1 = X[0][:,:,:,37:49]

                return torch.from_numpy(x1).permute(0,3,1,2)[0], torch.from_numpy(y1).permute(0,3,1,2)[0], time_row
            else:
                if idx%5 == 0:
                    X = super(NowcastGenerator1, self).__getitem__(idx1)  # N,L,W,49
                    x1 = X[0][:,:,:,:13]
                    y1 = X[0][:,:,:,13:25]
                elif idx%5 == 1:
                    X = super(NowcastGenerator1, self).__getitem__(idx1)  # N,L,W,49
                    x1 = X[0][:,:,:,6:19]
                    y1 = X[0][:,:,:,19:31]
                elif idx%5 == 2:
                    X = super(NowcastGenerator1, self).__getitem__(idx1)  # N,L,W,49
                    x1 = X[0][:,:,:,12:25]
                    y1 = X[0][:,:,:,25:37]
                elif idx%5 == 3:
                    X = super(NowcastGenerator1, self).__getitem__(idx1)  # N,L,W,49
                    x1 = X[0][:,:,:,18:31]
                    y1 = X[0][:,:,:,31:43]
                elif idx%5 == 4:
                    X = super(NowcastGenerator1, self).__getitem__(idx1)  # N,L,W,49
                    x1 = X[0][:,:,:,24:37]
                    y1 = X[0][:,:,:,37:49]

                return torch.from_numpy(x1).permute(0,3,1,2)[0], torch.from_numpy(y1).permute(0,3,1,2)[0]

        if self.data_amplifer == 3:
            idx1 = idx // 7
            if self.show_time:
                if idx%7 == 0:
                    X, time_row = super(NowcastGenerator1, self).__getitem__(idx1)  # N,L,W,49
                    x1 = X[0][:,:,:,:13]
                    y1 = X[0][:,:,:,13:25]
                elif idx%7 == 1:
                    X, time_row = super(NowcastGenerator1, self).__getitem__(idx1)  # N,L,W,49
                    x1 = X[0][:,:,:,4:17]
                    y1 = X[0][:,:,:,17:29]
                elif idx%7 == 2:
                    X, time_row = super(NowcastGenerator1, self).__getitem__(idx1)  # N,L,W,49
                    x1 = X[0][:,:,:,8:21]
                    y1 = X[0][:,:,:,21:33]
                elif idx%7 == 3:
                    X, time_row = super(NowcastGenerator1, self).__getitem__(idx1)  # N,L,W,49
                    x1 = X[0][:,:,:,12:25]
                    y1 = X[0][:,:,:,25:37]
                elif idx%7 == 4:
                    X, time_row = super(NowcastGenerator1, self).__getitem__(idx1)  # N,L,W,49
                    x1 = X[0][:,:,:,16:29]
                    y1 = X[0][:,:,:,29:41]
                elif idx%7 == 5:
                    X, time_row = super(NowcastGenerator1, self).__getitem__(idx1)  # N,L,W,49
                    x1 = X[0][:,:,:,20:33]
                    y1 = X[0][:,:,:,33:45]
                elif idx%7 == 6:
                    X, time_row = super(NowcastGenerator1, self).__getitem__(idx1)  # N,L,W,49
                    x1 = X[0][:,:,:,24:37]
                    y1 = X[0][:,:,:,37:49]

                return torch.from_numpy(x1).permute(0,3,1,2)[0], torch.from_numpy(y1).permute(0,3,1,2)[0], time_row
            else:
                if idx%7 == 0:
                    X = super(NowcastGenerator1, self).__getitem__(idx1)  # N,L,W,49
                    x1 = X[0][:,:,:,:13]
                    y1 = X[0][:,:,:,13:25]
                elif idx%7 == 1:
                    X = super(NowcastGenerator1, self).__getitem__(idx1)  # N,L,W,49
                    x1 = X[0][:,:,:,4:17]
                    y1 = X[0][:,:,:,17:29]
                elif idx%7 == 2:
                    X = super(NowcastGenerator1, self).__getitem__(idx1)  # N,L,W,49
                    x1 = X[0][:,:,:,8:21]
                    y1 = X[0][:,:,:,21:33]
                elif idx%7 == 3:
                    X = super(NowcastGenerator1, self).__getitem__(idx1)  # N,L,W,49
                    x1 = X[0][:,:,:,12:25]
                    y1 = X[0][:,:,:,25:37]
                elif idx%7 == 4:
                    X = super(NowcastGenerator1, self).__getitem__(idx1)  # N,L,W,49
                    x1 = X[0][:,:,:,16:29]
                    y1 = X[0][:,:,:,29:41]
                elif idx%7 == 5:
                    X = super(NowcastGenerator1, self).__getitem__(idx1)  # N,L,W,49
                    x1 = X[0][:,:,:,20:33]
                    y1 = X[0][:,:,:,33:45]
                elif idx%7 == 6:
                    X = super(NowcastGenerator1, self).__getitem__(idx1)  # N,L,W,49
                    x1 = X[0][:,:,:,24:37]
                    y1 = X[0][:,:,:,37:49]

                return torch.from_numpy(x1).permute(0,3,1,2)[0], torch.from_numpy(y1).permute(0,3,1,2)[0]

        elif self.data_amplifer == 4:
            idx1 = idx // 9
            if self.show_time:
                if idx%9 == 0:
                    X, time_row = super(NowcastGenerator1, self).__getitem__(idx1)  # N,L,W,49
                    x1 = X[0][:,:,:,:13]
                    y1 = X[0][:,:,:,13:25]
                elif idx%9 == 1:
                    X, time_row = super(NowcastGenerator1, self).__getitem__(idx1)  # N,L,W,49
                    x1 = X[0][:,:,:,3:16]
                    y1 = X[0][:,:,:,16:28]
                elif idx%9 == 2:
                    X, time_row = super(NowcastGenerator1, self).__getitem__(idx1)  # N,L,W,49
                    x1 = X[0][:,:,:,6:19]
                    y1 = X[0][:,:,:,19:31]
                elif idx%9 == 3:
                    X, time_row = super(NowcastGenerator1, self).__getitem__(idx1)  # N,L,W,49
                    x1 = X[0][:,:,:,9:22]
                    y1 = X[0][:,:,:,22:34]
                elif idx%9 == 4:
                    X, time_row = super(NowcastGenerator1, self).__getitem__(idx1)  # N,L,W,49
                    x1 = X[0][:,:,:,12:25]
                    y1 = X[0][:,:,:,25:37]
                elif idx%9 == 5:
                    X, time_row = super(NowcastGenerator1, self).__getitem__(idx1)  # N,L,W,49
                    x1 = X[0][:,:,:,15:28]
                    y1 = X[0][:,:,:,28:40]
                elif idx%9 == 6:
                    X, time_row = super(NowcastGenerator1, self).__getitem__(idx1)  # N,L,W,49
                    x1 = X[0][:,:,:,18:31]
                    y1 = X[0][:,:,:,31:43]
                elif idx%9 == 7:
                    X, time_row = super(NowcastGenerator1, self).__getitem__(idx1)  # N,L,W,49
                    x1 = X[0][:,:,:,21:34]
                    y1 = X[0][:,:,:,34:46]
                elif idx%9 == 8:
                    X, time_row = super(NowcastGenerator1, self).__getitem__(idx1)  # N,L,W,49
                    x1 = X[0][:,:,:,24:37]
                    y1 = X[0][:,:,:,37:49]

                return torch.from_numpy(x1).permute(0,3,1,2)[0], torch.from_numpy(y1).permute(0,3,1,2)[0], time_row
            else:
                if idx%9 == 0:
                    X = super(NowcastGenerator1, self).__getitem__(idx1)  # N,L,W,49
                    x1 = X[0][:,:,:,:13]
                    y1 = X[0][:,:,:,13:25]
                elif idx%9 == 1:
                    X = super(NowcastGenerator1, self).__getitem__(idx1)  # N,L,W,49
                    x1 = X[0][:,:,:,3:16]
                    y1 = X[0][:,:,:,16:28]
                elif idx%9 == 2:
                    X = super(NowcastGenerator1, self).__getitem__(idx1)  # N,L,W,49
                    x1 = X[0][:,:,:,6:19]
                    y1 = X[0][:,:,:,19:31]
                elif idx%9 == 3:
                    X = super(NowcastGenerator1, self).__getitem__(idx1)  # N,L,W,49
                    x1 = X[0][:,:,:,9:22]
                    y1 = X[0][:,:,:,22:34]
                elif idx%9 == 4:
                    X = super(NowcastGenerator1, self).__getitem__(idx1)  # N,L,W,49
                    x1 = X[0][:,:,:,12:25]
                    y1 = X[0][:,:,:,25:37]
                elif idx%9 == 5:
                    X = super(NowcastGenerator1, self).__getitem__(idx1)  # N,L,W,49
                    x1 = X[0][:,:,:,15:28]
                    y1 = X[0][:,:,:,28:40]
                elif idx%9 == 6:
                    X = super(NowcastGenerator1, self).__getitem__(idx1)  # N,L,W,49
                    x1 = X[0][:,:,:,18:31]
                    y1 = X[0][:,:,:,31:43]
                elif idx%9 == 7:
                    X = super(NowcastGenerator1, self).__getitem__(idx1)  # N,L,W,49
                    x1 = X[0][:,:,:,21:34]
                    y1 = X[0][:,:,:,34:46]
                elif idx%9 == 8:
                    X = super(NowcastGenerator1, self).__getitem__(idx1)  # N,L,W,49
                    x1 = X[0][:,:,:,24:37]
                    y1 = X[0][:,:,:,37:49]

                return torch.from_numpy(x1).permute(0,3,1,2)[0], torch.from_numpy(y1).permute(0,3,1,2)[0]



