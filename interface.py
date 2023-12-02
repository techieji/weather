from pyspeedy import Speedy, _speedy
from pyspeedy.error_codes import ERROR_CODES
from datetime import timedelta
import xarray as xr

class Model:
    TIME_STEP = timedelta(seconds=3600 * 24 / 36)
    def __init__(self, start_date, collect_data=True, interval=timedelta(days=1)):
        self.date = start_date
        self.model = Speedy()
        self.model.set_bc()
        self._data = []
        self.collect_data = collect_data
        self.interval = interval
        self._collection_date = self.date

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
        if self.collect_data and self._collection_date <= self.date:
            self._data.append(self.model.to_dataframe())
            while self._collection_date <= self.date:
                self._collection_date += self.interval