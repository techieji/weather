import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import tensor
from ncps.torch import LTC
from ncps.wirings import AutoNCP
import xarray as xr
import numpy as np

from dataclasses import dataclass
import os.path

rng = np.random.default_rng(seed=1)

ONNX_EXPORT = False

CONST_TABLE = None
const_sel = None
_const_sel = None

RAW_DATA = None
DATA = None

PRED_VARS = 'uwind vwind air shum'.split()    # Add phi ps
VAR_N = len(PRED_VARS)
LEVELS = 17
CONSTANTS = 'orog lsm alb vegh vegl'.split()

N_CONVS = 1
GLOBAL_TILE = 9  # Like a radius
CONVOL_TILE = 3
OUTPUT_TILE = 3
TIME_LENGTH = 10

LTC_SPARSITY = 0.5

PRED_N = 28

fields = len(PRED_VARS) * LEVELS + len(CONSTANTS)

@dataclass
class PrognosticVars:
    vars: xr.Dataset
    pres: xr.Dataset

def load_data(data_loc='~/data', force=False):
    global DATA, RAW_DATA
    if DATA is None or force:
        RAW_DATA = PrognosticVars(
            vars=xr.combine_by_coords([
                xr.open_mfdataset(os.path.join(data_loc, 'air.*.nc')),
                xr.open_mfdataset(os.path.join(data_loc, 'shum.*.nc')),
                xr.open_mfdataset(os.path.join(data_loc, 'uwnd.*.nc')),
                xr.open_mfdataset(os.path.join(data_loc, 'vwnd.*.nc')),
            ], combine_attrs='drop_conflicts'),
            pres=xr.open_mfdataset(os.path.join(data_loc, 'pres.sfc.*.nc')),
        )
        DATA = PrognosticVars(
            vars=RAW_DATA.vars.interp(lon=CONST_TABLE.lon, lat=CONST_TABLE.lat),
            pres=RAW_DATA.pres.interp(lon=CONST_TABLE.lon, lat=CONST_TABLE.lat)
        )   # TODO find new boundary conditions or upsample CONST_TABLE
    return DATA

def load_const_table(location='/pySPEEDY/pyspeedy/data/example_bc.nc', force=False):
    global CONST_TABLE
    if CONST_TABLE is None or force:
        data = xr.load_dataset(location)
        CONST_TABLE = data[CONSTANTS]
    return CONST_TABLE

def load_all():
    load_const_table()
    print("Constants loaded")
    load_data()
    print("Data loaded")

def is_data_loaded():
    return CONST_TABLE is not None and DATA is not None

def mod_iter(i, mod):
    return map(lambda x: x % mod, i)

def slice_pos(df, loni, lati):
    posw = GLOBAL_TILE
    lonr = [x % len(df.lon) for x in range(loni-posw, loni+posw-1)]
    latr = [x % len(df.lat) for x in range(lati-posw, lati+posw-1)]
    return df.sel(lon=df.lon[lonr], lat=df.lat[latr])

def slice_time(df, t):
    tlen = TIME_LENGTH + PRED_N
    return df.sel(time=df.time[t:t+tlen])

def sel_const(loni, lati):
    global const_sel, _const_sel
    const_sel = slice_pos(CONST_TABLE, loni, lati).to_dataarray().transpose()
    _const_sel = torch.asarray(const_sel.to_numpy())
    return const_sel

def to_tensors(arr):    # time x vars x lon x lat x level
    return torch.nan_to_num(torch.asarray(arr.to_numpy())).to(torch.float32)

def random_pos(): # raw values: lon, lat, time
    posw = GLOBAL_TILE
    tlen = TIME_LENGTH
    df = load_data().vars
    loni = rng.integers(0, len(df.lon))
    lati = rng.integers(0, len(df.lat))
    ti = rng.integers(0, len(df.time) - tlen)
    return loni, lati, ti

def random_slice():
    loni, lati, ti = random_pos()
    sel_const(loni, lati)
    tens = slice_pos(slice_time(DATA.vars, ti), loni, lati).to_dataarray()
    return tens.transpose('time', 'variable', 'lon', 'lat', 'level')

class AdaptiveMaxPool3dCustom(nn.Module):     # For ONNX compatibility
    def __init__(self, output_size):
        super().__init__()
        self.output_size = np.array(output_size)

    def forward(self, x: torch.Tensor):
        stride_size = np.floor(np.array(x.shape[-3:]) / self.output_size).astype(np.int32)
        kernel_size = np.array(x.shape[-3:]) - (self.output_size - 1) * stride_size
        avg = nn.MaxPool3d(kernel_size=list(kernel_size), stride=list(stride_size))
        x = avg(x)
        return x

class VarEncoder(nn.Module):
    def __init__(self, inp, out):
        super().__init__()
        self.seq = nn.Sequential(
            nn.Linear(inp, inp),
            nn.Sigmoid(),
            nn.Linear(inp, out)
        )

    def forward(self, x):
        return self.seq(x)

class CrossConnector(nn.Module):
    def __init__(self, tile_size, ntiles, interm, output, tile_dim=3):
        super().__init__()
        # batch is optional
        # batch x ntiles x [[tile_size]:tile_dim] => batch x interm => batch x output
        self.seq = nn.Sequential(
                nn.Flatten(-tile_dim, -1),
                nn.Linear(tile_size, tile_size),   # TODO: replace with VarEncoder
                nn.Sigmoid(),
                nn.Linear(tile_size, (tile_size + interm) // 2),
                nn.Sigmoid(),
                nn.Linear((tile_size + interm) // 2, interm),
                nn.Sigmoid(),
                # Tiles turned into interm
                nn.Flatten(-2, -1),
                nn.Linear(ntiles*interm, ntiles*interm),   # TODO: replace with VarEncoder
                nn.Sigmoid(),
                nn.Linear(ntiles*interm, (ntiles*interm + output) // 2),
                nn.Sigmoid(),
                nn.Linear((ntiles*interm + output) // 2, output)
        )

    def forward(self, x):
        # x: vars x (pos)
        return self.seq(x)

class RegionEncoder(nn.Module):   # TODO: optimize for functional programming
    def __init__(self):
        # Channels:
        #   Prognostic vars: u, v, t, q, phi, ps
        #   Time-invarient: orog, lsm(?), alb, veghigh, veglow
        
        super().__init__()

        self.dropout = nn.Dropout()
        # 1. Combine each level into the constants => vars x (lon x lat x level)
        self.encs  = [VarEncoder(LEVELS + len(CONSTANTS), LEVELS) for _ in PRED_VARS]
        # 2. Mix vars (applied by lon, lat, t)     => ps-vars x (lon x lat x level)
        self.mixer = VarEncoder(VAR_N, VAR_N)
        # 3. Apply n 3D convolution seperately     => ps-vars x (lon x lat x level) [mostly]
        self.convs = [nn.Conv3d(VAR_N, VAR_N, CONVOL_TILE) for _ in range(N_CONVS)]
        # 4. Pool for each pseudo-var              => ps-vars x (output x output x output)
        if ONNX_EXPORT:
            self.pool = AdaptiveMaxPool3dCustom(OUTPUT_TILE)
        else:
            self.pool = nn.AdaptiveMaxPool3d(OUTPUT_TILE)
        # 5. Fully connected linear layers         => ps-vars x level [homogenous]
        self.layer = CrossConnector(OUTPUT_TILE**3, VAR_N, VAR_N*LEVELS, VAR_N*LEVELS)

    def forward(self, x):    # TODO: make all dimensions negative to allow efficient application
        # TODO: allow batch to be optional
        # (pos) = lon x lat x level
        # x: batch x vars x (pos)
        x = self.dropout(x)
        # vs: batch x (pos) x vars
        vs = torch.stack(
            [enc(torch.cat((
                var, _const_sel.unsqueeze(0).repeat(*var.shape[:-3], *[1]* 3)   # 3 is num of dims of _const_sel  FIXME check if this works
            ), dim=-1)) for var, enc in zip(x.unbind(1), self.encs)]
        ).transpose(0, 1).movedim((-1, -2, -3), (-2, -3, -4))
        # pvs: batch x pvars x (pos)
        pvs = torch.movedim(self.mixer(vs), (-2, -3, -4), (-1, -2, -3))
        for conv in self.convs:
            pvs = conv(pvs)
        pvs = self.pool(pvs)
        # returns (1D): batch x level * vars
        return self.layer(pvs)

class LiquidOperator(nn.Module):
    def __init__(self):
        super().__init__()

        # 6. Fully connected preprocessing layer   => time x vars * level
        self.preproc = VarEncoder(VAR_N*LEVELS, VAR_N*LEVELS)
        # 7. LTC                                   => time x vars x level
        self.wiring = AutoNCP(3*LEVELS, LEVELS, sparsity_level=LTC_SPARSITY)  # TODO: investigate whether different wirings need to be created
        self.ltcs = [LTC(LEVELS, self.wiring) for _ in range(VAR_N)]
        # 8. Fully connected postprocessing layers => time x vars * level
        self.postproc = CrossConnector(LEVELS, VAR_N, VAR_N*LEVELS, VAR_N*LEVELS, tile_dim=1)

    @property
    def pred_n(self): return PRED_N

    def forward(self, x):
        # xs: vars x (time x level)
        xs = self.preproc(x).reshape((-1, VAR_N, LEVELS)).unbind(1)
        # r: time x vars x level
        rs = torch.stack(
            [ltc(torch.cat((ts, ts[-1].repeat(self.pred_n, 1))))[0] for ltc, ts in zip(self.ltcs, xs)]   # ts: time x level
        ).transpose(0, 1)
        return self.postproc(rs)

def persistence_model(x):
    # lt: var x level
    lt = x[-1,:,GLOBAL_TILE,GLOBAL_TILE,:]
    return lt.flatten(1).repeat(TIME_LENGTH + PRED_N, 1, 1).flatten(1)

class WeatherPredictor(nn.Module):
    def __init__(self):
        super().__init__()
        self.re = RegionEncoder()
        self.lo = LiquidOperator()
        self.base_model = persistence_model

    def forward(self, x):
        return self.lo(self.re(x)) + self.base_model(x)
