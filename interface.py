from pyspeedy import Speedy, _speedy
from pyspeedy.error_codes import ERROR_CODES
from datetime import timedelta
import xarray as xr
import numpy as np
from scipy.special import roots_legendre
import os.path
from dataclasses import dataclass

@dataclass
class PrognosticVars:
    vars: xr.Dataset
    pres: xr.Dataset

class Dycore:
    TIME_STEP = timedelta(seconds=3600 * 24 / 36)
    LOADED_DATA = None
    SIGMA_LEVELS = np.array([0.025, 0.095, 0.20, 0.34, 0.51, 0.685, 0.835, 0.95])
    def __init__(self, start_date, end_date, collect_data=True, interval=timedelta(days=1)):
        self.start_date = start_date
        self.date = start_date
        self.end_date = end_date
        self.model = Speedy(start_date=start_date, end_date=end_date)
        self.model.set_bc()
        self._data = []
        self.collect_data = collect_data
        self.interval = interval
        self._collection_date = self.date

    @property
    def running(self):
        return self.date < self.end_date

    @property
    def data(self):
        return xr.concat(self._data, dim='time')
    
    def __getattr__(self, attr):
        try:
            return self.model[attr]
        except KeyError:
            raise AttributeError(f"Attribute not found: {attr}")

    def step(self):
        error = _speedy.step(self.model._state_cnt, self.model._control_cnt)
        if error < 0:
            raise RuntimeError(f"Speedy: {ERROR_CODES[error]}")
        self.date += Model.TIME_STEP
        self.model.current_date += Model.TIME_STEP
        if self.collect_data and self._collection_date <= self.date:
            self._data.append(self.model.to_dataframe())
            while self._collection_date <= self.date:
                self._collection_date += self.interval

    @staticmethod
    def gaussian_lats(n):   # Written by ChatGPT, TODO: CHECK!!!
        roots, _ = roots_legendre(n)
        return roots * 90

    def set_data(self, **kwargs):
        data = Dycore.load_data(**kwargs)
        vs = data.vars.interp(
            lat=Dycore.gaussian_lats(48),
            lon=np.linspace(0, 360, 96),
            level=Dycore.SIGMA_LEVELS * 1000,    # This is not correct, adjust for surface pressure FIXME
            time=self.date
        ).transpose()
        pres = data.pres.interp(
            lat=Dycore.gaussian_lats(48),
            lon=np.linspace(0, 360, 96),
            time=self.date
        ).transpose()

        #print(self.model['t_grid'])
        print(vs.air.to_numpy())

        # self.model['t_grid'] = vs.air
        # self.model['u_grid'] = vs.uwnd
        # self.model['v_grid'] = vs.vwnd
        # self.model['q_grid'] = vs.shum
        # self.model['ps_grid'] = pres.pres
        self.model.grid2spectral()

    @classmethod
    def load_data(klass, data_loc='~/data', force=False):
        if Dycore.LOADED_DATA is None or force:
            Dycore.LOADED_DATA = PrognosticVars(
                vars=xr.combine_by_coords([
                    xr.open_mfdataset(os.path.join(data_loc, 'air.*.nc')),
                    xr.open_mfdataset(os.path.join(data_loc, 'shum.*.nc')),
                    xr.open_mfdataset(os.path.join(data_loc, 'uwnd.*.nc')),
                    xr.open_mfdataset(os.path.join(data_loc, 'vwnd.*.nc')),
                ], combine_attrs='drop_conflicts'),
                pres=xr.open_mfdataset(os.path.join(data_loc, 'pres.sfc.*.nc')),
            )
        return Dycore.LOADED_DATA

Model = Dycore
