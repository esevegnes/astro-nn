### Librairies
from keras.src import ops
from keras.src.metrics import reduction_metrics
from keras.src.losses.loss import squeeze_or_expand_to_same_rank
import pandas as pd
import numpy as np

import h5py
from datetime import datetime
import torch
import torch.nn as nn
from torch.utils.data import Dataset

from pathlib import Path
from torch.profiler import profile, tensorboard_trace_handler, ProfilerActivity, schedule
from sklearn.metrics import mean_squared_error, max_error, r2_score
from torch.utils.data import DataLoader, random_split

### Pysical constants
e = 1.602176634e-19
m_i = 1.67262192369e-27
m_e = 9.1093837015e-31
### Functions
def E_mhd(df):
    '''
    [B] = nT
    [u] = km/s
    [E] = µV/m => mV/m
    '''
    dg = pd.DataFrame(index=df.index,columns=['ex','ey','ez'])
    dg['ex']=(df['by']*df['uz']-df['uy']*df['bz'])/1e3
    dg['ey']=(df['bz']*df['ux']-df['uz']*df['bx'])/1e3 
    dg['ez']=(df['bx']*df['uy']-df['ux']*df['by'])/1e3
    return dg

def E_hall(df):
    '''
    [B] = nT
    [j] = A/m^2
    [n] = cm^-3
    [E] = *1e-15 V/m => 1e-12 mV/m
    '''
    dg = pd.DataFrame(index=df.index,columns=['ex','ey','ez'])
    dg['ex']= ((df['jy']*df['bz']-df['by']*df['jz'])/(e*df['e_density']))/1e12
    dg['ey']= ((df['jz']*df['bx']-df['bz']*df['jx'])/(e*df['e_density']))/1e12
    dg['ez']= ((df['jx']*df['by']-df['bx']*df['jy'])/(e*df['e_density']))/1e12
    return dg

### Class
#### Metrics
class MaxError(reduction_metrics.Mean):
    
    def __init__(self, scaler, device, name="max_error", dtype=None):
        super().__init__(name, dtype=dtype)
        # Metric should be minimized during optimization.
        self._direction = "down"
        self.scaler = scaler
        self.device = device

    def update_state(self, y_true, y_pred, sample_weight=None):
        """Accumulates root mean squared error statistics.

        Args:
            y_true: The ground truth values.
            y_pred: The predicted values.
            sample_weight: Optional weighting of each example. Can
                be a `Tensor` whose rank is either 0, or the same rank as
                `y_true`, and must be broadcastable to `y_true`.
                Defaults to `1`.

        Returns:
            Update op.
        """
        y_true = ops.convert_to_tensor(self.scaler.untransform(torch.zeros((y_true.size()[0],self.scaler.__len__()-3)).to(self.device),y_true)[1], self._dtype)
        y_pred = ops.convert_to_tensor(self.scaler.untransform(torch.zeros((y_pred.size()[0],self.scaler.__len__()-3)).to(self.device),y_pred)[1], self._dtype)
        y_true, y_pred = squeeze_or_expand_to_same_rank(y_true, y_pred)
        max_err = ops.abs(ops.max(y_pred - y_true,axis=0))
        self.max_err = max_err
        return super().update_state(max_err, sample_weight=sample_weight)

    def result(self):
        return(self.max_err)
    
    def get_config(self):
        return {"scaler":self.scaler,"device":self.device}
    
class MSE(reduction_metrics.Mean):
    
    def __init__(self, scaler, device, name="MSE", dtype=None):
        super().__init__(name, dtype=dtype)
        # Metric should be minimized during optimization.
        self._direction = "down"
        self.scaler = scaler
        self.device = device

    def update_state(self, y_true, y_pred, sample_weight=None):
        """Accumulates root mean squared error statistics.

        Args:
            y_true: The ground truth values.
            y_pred: The predicted values.
            sample_weight: Optional weighting of each example. Can
                be a `Tensor` whose rank is either 0, or the same rank as
                `y_true`, and must be broadcastable to `y_true`.
                Defaults to `1`.

        Returns:
            Update op.
        """
        y_true = ops.convert_to_tensor(self.scaler.untransform(torch.zeros((y_true.size()[0],self.scaler.__len__()-3)).to(self.device),y_true)[1], self._dtype)
        y_pred = ops.convert_to_tensor(self.scaler.untransform(torch.zeros((y_pred.size()[0],self.scaler.__len__()-3)).to(self.device),y_pred)[1], self._dtype)
        y_true, y_pred = squeeze_or_expand_to_same_rank(y_true, y_pred)
        mse = ops.mean(ops.square(y_pred - y_true),axis=0)
        self.mse = mse
        return super().update_state(mse, sample_weight=sample_weight)

    def result(self):
        return(self.mse)
    
    def get_config(self):
        return {"scaler":self.scaler,"device":self.device}

class PCC(reduction_metrics.Mean):
    
    def __init__(self, name="PCC", dtype=None):
        super().__init__(name, dtype=dtype)
        # Metric should be maximized during optimization.
        self._direction = "up"

    def update_state(self, y_true, y_pred, sample_weight=None):
        """Accumulates root mean squared error statistics.

        Args:
            y_true: The ground truth values.
            y_pred: The predicted values.
            sample_weight: Optional weighting of each example. Can
                be a `Tensor` whose rank is either 0, or the same rank as
                `y_true`, and must be broadcastable to `y_true`.
                Defaults to `1`.

        Returns:
            Update op.
        """
        y_true = ops.convert_to_tensor(y_true, self._dtype)
        y_pred = ops.convert_to_tensor(y_pred, self._dtype)
        y_true, y_pred = squeeze_or_expand_to_same_rank(y_true, y_pred)
        pcc = torch.tensor([torch.corrcoef(torch.stack((y_true[:,i],y_pred[:,i])))[0,1] for i in range(3)]) 
        self.pcc = pcc
        return super().update_state(pcc, sample_weight=sample_weight)
    
        #y_true, y_pred = torch.randn((10,3)), torch.randn((10,3))
        #torch.tensor([torch.corrcoef(torch.stack((y_true[:,i],y_pred[:,i])))[0,1] for i in range(3)]) - torch.tensor([np.corrcoef(y_true[:,i], y_pred[:,i])[0,1] for i in range(3)])    
        #tensor([-1.5264e-08, -3.3849e-08,  3.4143e-08], dtype=torch.float64) ! Validation that numpy pcc (with 2 inputs) is the same that torch pcc when stacking arrays (only one input)
    
    def result(self):
        return(self.pcc)

#### Dataset
class MMS_Dataset(Dataset):
    def __init__(self,sat=str,data_path=str,t1=datetime,t2=datetime,density_threshold=float,OHM=bool,XYZ=False):
        
        m_i = 1.67262192369e-27
        m_e = 9.1093837015e-31

        file = h5py.File(f'{data_path}','r')
        bursts = list(file[f"{sat}"].keys())
        file.close()
        bursts = pd.DataFrame([datetime.strptime(burst, f'%Y_%m_%dT%H_%M_%S') for burst in bursts])
        bursts = bursts.where((t1 < bursts)&(bursts < t2)).dropna()
        bursts = [datetime.strftime(event,'%Y_%m_%dT%H_%M_%S') for event in bursts[0]]

        df = pd.concat([pd.read_hdf(f'{data_path}',key=f"{sat}/{event}") for event in bursts]).dropna()

        ## Calculating velocity, and droping small density data
        df['ux']=(m_i*df['vx_i']+m_e*df['vx_e'])/(m_i+m_e)
        df['uy']=(m_i*df['vy_i']+m_e*df['vy_e'])/(m_i+m_e)
        df['uz']=(m_i*df['vz_i']+m_e*df['vz_e'])/(m_i+m_e)
        df = df.where(df['e_density']>density_threshold).dropna()
        if OHM:
            df[['ex_mhd','ey_mhd','ez_mhd']] = E_mhd(df)
            df[['ex_hall','ey_hall','ez_hall']] = E_hall(df)
            input_features = ['bx', 'by', 'bz',
                    'jx', 'jy', 'jz',
                    'ux', 'uy', 'uz',
                    'e_density',
                    'ex_mhd','ey_mhd','ez_mhd',
                    'ex_hall','ey_hall','ez_hall',
                    ]
            output_targets = ['ex','ey','ez']
        else:
        ## Input / Output wanted
            input_features = ['bx', 'by', 'bz',
                        'jx', 'jy', 'jz',
                        'ux', 'uy', 'uz',
                        'e_density']
            output_targets = ['ex','ey','ez']
        if XYZ:
            self.xyz = df[['x','y','z']]
        df = df.drop(df.columns.drop(input_features+output_targets),axis=1) #drop useless data
        df = df[input_features + output_targets] #reorder columns for input to the left and output to the right

        #self.df = df
        self.columns = df.columns
        self.index = df.index
        self.values = torch.tensor(df.values)
        self.bursts = bursts

    def __len__(self):
        return len(self.values)

    def __getitem__(self, index):
        return(self.values[index])

    def to_dataframe(self):
        return(pd.DataFrame(data=self.values,index=self.index,columns=self.columns))

    def exclude(self,t1_ex,t2_ex):
        # Build boolean mask (works for pandas Index or numpy array)
        keep_mask = (self.index < t1_ex) | (self.index > t2_ex)

        # Ensure numpy boolean array
        keep_mask_np = np.asarray(keep_mask, dtype=bool)

        # Apply mask to tensor and index
        self.values = self.values[torch.from_numpy(keep_mask_np)]
        self.index = self.index[keep_mask_np]
        return self
#### Scalers
class Standard_Scaler(Dataset):
    def __init__(self,dataset):
        self.means = dataset.dataset.values.mean(axis=0)
        self.stds = dataset.dataset.values.std(axis=0)

    def __len__(self):
        return len(self.means)

    def __getitem__(self, index):
        return(self.means[index]) 
    
    def to(self,device):
        self.means = self.means.to(device)
        self.stds = self.stds.to(device)

    def transform(self, dataset):
        data = dataset.dataset.values
        data_scaled = (data - self.means) / self.stds
        return(data_scaled[:,:-3],data_scaled[:,-3:])
    
    def untransform(self, X, y):
        return(X*(self.stds[:-3]) + self.means[:-3],y*(self.stds[-3:]) + self.means[-3:])

    def get_config(self):
        return {"means":self.means,"stds":self.stds}
    

class MinMax_Scaler(Dataset):
    def __init__(self,dataset):
        self.mins = dataset.dataset.values.min(axis=0)[0]
        self.maxs = dataset.dataset.values.max(axis=0)[0]

    def __len__(self):
        return len(self.mins)

    def to(self,device):
        self.mins = self.mins.to(device)
        self.maxs = self.maxs.to(device)

    def __getitem__(self, index):
        return(self.means[index]) 
    
    def transform(self, dataset):
        data = dataset.dataset.values
        data_scaled = (data - self.mins) / (self.maxs - self.mins)
        return(data_scaled[:,:-3],data_scaled[:,-3:])

    def untransform(self, X, y):
        return(X * (self.maxs[:-3] - self.mins[:-3]) + self.mins[:-3],y * (self.maxs[-3:] - self.mins[-3:]) + self.mins[-3:])

    def get_config(self):
        return {"mins":self.mins,"maxs":self.maxs}