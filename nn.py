import torch
import torch.nn as nn
from torch import tensor
from ncps.torch import LTC
from ncps.wirings import AutoNCP
import xarray as xr
import numpy as np

from dataclasses import dataclass
import os.path

rng = np.random.default_rng(seed=1)

CONST_TABLE = None
const_sel = None
_const_sel = None

RAW_DATA = None
DATA = None

PRED_VARS = 'uwind vwind air shum'.split()    # phi ps
VAR_N = len(PRED_VARS)
LEVELS = 8
CONSTANTS = 'orog lsm alb vegh vegl'.split()

N_CONVS = 1
GLOBAL_TILE = 20
CONVOL_TILE = 10
OUTPUT_TILE = 3
TIME_LENGTH = 20

LTC_SPARSITY = 0.8

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
        )
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

def slice_pos(df, loni, lati):
    posw = GLOBAL_TILE
    return df.sel(lon=df.lon[loni-posw:loni+posw], lat=df.lat[lati-posw:lati+posw])

def slice_time(df, t):
    tlen = TIME_LENGTH
    return df.sel(time=df.time[t:t+tlen])

def sel_const(loni, lati):
    global const_sel, _const_sel
    const_sel = slice_pos(CONST_TABLE, loni, lati).to_dataarray().transpose()
    _const_sel = torch.asarray(const_sel.to_numpy()).unbind()
    return const_sel

def to_tensor(df, swap_from=0, swap_to=1):    # time x vars x lon x lat x level
    return torch.asarray(np.moveaxis(
        df.to_dataarray().to_numpy(),
        swap_from,
        swap_to
    ))

def random_pos(): # raw values: lon, lat, time
    posw = GLOBAL_TILE
    tlen = TIME_LENGTH
    df = load_data().vars
    loni = rng.integers(posw, len(df.lon) - posw)
    lati = rng.integers(posw, len(df.lat) - posw)
    ti = rng.integers(0, len(df.time) - tlen)
    return loni, lati, ti

def random_slice():
    loni, lati, ti = random_pos()
    sel_const(loni, lati)
    return slice_pos(slice_time(DATA.vars, ti), loni, lati)

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
    def __init__(self, tile_size, ntiles, interm, output):
        super().__init__()
        self.seq = nn.Sequential(
                nn.Flatten(),
                nn.Linear(tile_size, tile_size),
                nn.Sigmoid(),
                nn.Linear(tile_size, (tile_size + interm) // 2),
                nn.Sigmoid(),
                nn.Linear((tile_size + interm) // 2, interm),
                nn.Sigmoid(),
                # Tiles turned into interm
                nn.Flatten(0, -1),
                nn.Linear(ntiles*interm, ntiles*interm),
                nn.Sigmoid(),
                nn.Linear(ntiles*interm, (ntiles*interm + output) // 2),
                nn.Sigmoid(),
                nn.Linear((ntiles*interm + output) // 2, output)
        )

    def forward(self, x):
        # x: vars x (pos)
        return self.seq(x)

class RegionEncoder(nn.Module):
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
        self.convs = [[nn.Conv3d(VAR_N, VAR_N, CONVOL_TILE, groups=VAR_N) for _ in PRED_VARS] for _ in range(N_CONVS)]
        # 4. Pool for each pseudo-var              => ps-vars x (output x output x output)
        self.pool = nn.AdaptiveMaxPool3d(OUTPUT_TILE)
        # 5. Fully connected linear layers         => ps-vars x level [homogenous]
        self.layer = CrossConnector(OUTPUT_TILE**3, VAR_N, VAR_N*LEVELS, VAR_N*LEVELS)

    def forward(self, x):
        # (pos) = lon x lat x level
        # x: vars x (pos)
        x = self.dropout(x)
        # vs: (pos) x vars
        vs = torch.movedim(torch.stack(
            [enc(torch.stack([var] + _const_sel)) for var, enc in zip(torch.unbind(x), self.encs)]
        ), (1, 2, 3), (0, 1, 2))
        # pvs: pvars x (pos)
        pvs = torch.movedim(self.mixer(vs), (0, 1, 2), (1, 2, 3))
        for convs in self.convs:
            pvs = torch.stack([conv(var) for var, conv in zip(torch.unbind(pvs), convs)])
        pvs = self.pool(pvs)
        return self.layer(pvs)

class LiquidOperator(nn.Module):
    def __init__(self):
        super().__init__()

        # 6. LTC                                   => vars x level
        self.wiring = AutoNCP(3*VAR_N*LEVELS, VAR_N*LEVELS, sparsity_level=LTC_SPARSITY)
        self.ltc = LTC(VAR_N*LEVELS, wiring)

    def forward(self, x):
        pass
