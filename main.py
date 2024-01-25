import nn
import torch
import torch.nn.functional as F
import numpy as np
from time import time

import sys
import argparse
from dataclasses import dataclass
from collections import namedtuple
import random
import glob

model = None
validate = None

loss = torch.nn.L1Loss()

def setup(do_model=True, do_data=True):
    global model, validate, optim, scheduler
    if do_data:
        nn.load_all()
    if do_model:
        model = nn.WeatherPredictor()
        validate = Validator()

        optim = torch.optim.Adam(model.parameters(), lr=LR)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optim)

LR = 0.07    # LR: 0.03

def calc_loss(sl=None, model=None, needs_grad=True, train_mode=True, n=1):
    if model is None: model = globals()['model']   # Bad code on my part
    try:
        model.train(train_mode)
    except:
        pass
    ret = 0
    pcc = 0
    for _ in range(n):
        if sl is None:
            _sl = nn.random_slice()
            sl = nn.to_tensors(_sl)
        warmup = sl[:nn.TIME_LENGTH]
        warmup.requires_grad_(needs_grad)
        r = model(warmup)
        value = sl[nn.TIME_LENGTH:,:,nn.GLOBAL_TILE,nn.GLOBAL_TILE,:].flatten(1)
        rvalue = r[nn.TIME_LENGTH:]
        anom = nn._anom_sel.flatten(1)[nn.TIME_LENGTH:]
        ret += loss(value, rvalue)
        pcc += nn.anomaly_correlation(value, rvalue, anom)
    return ret/n, pcc/n

def train(n=1):
    l, pcc = calc_loss(n=n)
    optim.zero_grad()
    l.backward()
    optim.step()
    return l.sum().item(), pcc.sum().item()

class Validator:
    def __init__(self, n=10):
        print('=> Setting up validator...')
        self.n = n
        self.slices = []
        self._const_sels = []
        self._anom_sels = []
        for _ in range(n):
            self.slices.append(nn.to_tensors(nn.random_slice()))
            self._const_sels.append(nn._const_sel)
            self._anom_sels.append(nn._anom_sel)
        print('Validator is set up')

    def _switch_to(self, n):
        nn._const_sel = self._const_sels[n]
        nn._anom_sel = self._anom_sels[n]
        return self.slices[n]

    def __call__(self, model=model):
        with torch.no_grad():
            losses = []
            pcces = []
            for i in range(self.n):
                sl = self._switch_to(i)
                l, pcc = calc_loss(sl, model, False, False)
                losses.append(l)
                pcces.append(pcc)
            return sum(losses), sum(pcces)/len(pcces)

def data_saver(n, folder=nn.DATAENTRY_FOLDER):
    i = 1
    c = 0
    while c < n:
        try:
            with open(f'{folder}/dataentry-{i}.pt', 'xb') as f:
                torch.save(nn.DataEntry.get_current_entry(nn.random_slice_slow()), f)
        except FileExistsError:
            i += 1
        else:
            print('Saved: ', i)
            i += 1
            c += 1

EpochData = namedtuple('EpochData', 'train_loss val_loss pcc')

def random_word():
    vowels = 'aeiuo'
    consonents = 'qwrtypsdfghjklzxcvbnm'
    digraphs = 'sc ng ch ph sh it th wh ci qu'.split()
    conson_like = list(consonents) + digraphs
    l = 5
    return ''.join(random.choice(conson_like if i % 2 == 0 else vowels) for i in range(l))

@dataclass
class ModelData:
    epoch_num: int = 0
    constructor: str = ''
    big: bool = False
    learning_rate: float = 0
    base_data: EpochData = None
    base_model: str = 'persistence'    # Probably going to be fixed
    epoch_data: list[EpochData] = None
    model: torch.nn.Module = None

    def add_epoch_data(self, train_loss, val_loss, pcc):
        if self.epoch_data is None:
            self.epoch_data = []
        self.epoch_data.append(EpochData(train_loss, val_loss, pcc))
        self.epoch_num += 1

    def set_base_data(self, loss, pcc):
        self.base_data = EpochData(None, loss, pcc)

    def get_var(self, var: str):
        return [getattr(x, var) for x in self.epoch_data]

    def save(self, folder='models'):
        name = random_word() + '_' + random_word()
        print(f'=> Saving model {name}')
        torch.save(self, f'{folder}/model-{name}.pt')
        return name

    @staticmethod
    def load(name, folder='models'):
        return torch.load(f'{folder}/model-{name}.pt')

    @staticmethod
    def load_all(folder='models', raise_errors=True):
        for file in glob.iglob(f'{folder}/model-*.pt'):
            name = file.split('-')[-1][:-3]
            try:
                yield ModelData.load(name)
            except:
                if raise_errors:
                    raise
                print(f'{name} is defective... somehow')

    @staticmethod
    def get_representatives(folder='models', raise_errors=True):
        # Mainly four categories: lstm-forked, lstm-big, ltc-forked, ltc-big
        lstm_fork, lstm_big, ltc_fork, ltc_big = [], [], [], []
        for model in ModelData.load_all(folder, raise_errors):
            if model.constructor == 'lstm':
                if model.big: lstm_big.append(model)
                else: lstm_fork.append(model)
            else:
                if model.big: ltc_big.append(model)
                else: ltc_fork.append(model)

def _main(epoch_count, stop_limit):
    last_val_loss = np.inf
    stop_n = 0
    for i in range(epoch_count):
        if stop_n > stop_limit:
            print('Training stopped')
            break
        try:
            val_loss, val_pcc = validate()
            val_loss = val_loss.sum()
            val_pcc = val_pcc.sum()
            train_loss, train_pcc = train(n=1)
            yield (val_loss.item(), train_loss, val_pcc.item())
            if val_loss - last_val_loss > 0:
                stop_n += 1
            else:
                stop_n = 0
            last_val_loss = val_loss
            scheduler.step(val_loss)
        except Exception as e:
            print(f'Error ({i}):', e)

def main(save, *args, **kwargs):
    print('=> Setting up data storage system')
    data = ModelData()
    data.constructor = nn.CONSTRUCTOR
    data.big = nn.BIG
    data.learning_rate = LR
    # TODO: set up constructor, big, learning rate
    print('=> Calculating base loss')
    base_loss, base_pcc = validate(model=nn.persistence_model)
    base_loss = base_loss.item()
    base_pcc = base_pcc.item()
    data.set_base_data(base_loss, base_pcc)
    print('Base loss:', base_loss)
    print('Base PCC :', base_pcc)
    print('Epoch\t Loss \t\t Diff \t\t PCC ')
    print('-----\t------\t\t------\t\t-----')
    for i, loss in enumerate(_main(*args, **kwargs)):
        # print(i, loss[0], base_loss - loss[0], loss[2], sep='\t')
        print(f'{i}\t{loss[0]:.6f}\t{base_loss - loss[0]:.6f}\t{loss[2]:.6f}')
        data.add_epoch_data(loss[1], loss[0], loss[2])
    print('Final diff:', base_loss - loss[0])
    if (base_loss - loss[0] > 0) or save:
        print('success!')
        data.model = model
        data.save()
    else:
        print('fail.')

if __name__ == '__main__':
    parser = argparse.ArgumentParser(prog='weather-trainer')
    parser.add_argument('-e', '--epoch-count', type=int, default=100)
    parser.add_argument('-s', '--stop-num', type=int, default=5)
    parser.add_argument('-t', '--test-forwards', action='store_true')
    parser.add_argument('-d', '--data-cache', action='store_true')
    parser.add_argument('-O', '--optimized', action='store_true')
    parser.add_argument('-c', '--save', action='store_true')
    parser.add_argument('-m', '--model-type', choices=['lstm', 'ltc'], default='ltc')
    parser.add_argument('-b', '--big-model', action='store_true')
    args = parser.parse_args()
    nn.CONSTRUCTOR = args.model_type
    nn.BIG = args.big_model
    if args.optimized:
        nn.random_slice = nn.random_slice_fast
    if args.test_forwards:
        diam = 2*nn.GLOBAL_TILE - 1
        nn._const_sel = torch.zeros((diam, diam, nn.VAR_C))
        sl = torch.zeros((nn.TIME_LENGTH, nn.VAR_N, diam, diam, nn.LEVELS))
        print('Starting test')
        model = nn.WeatherPredictor()
        model(sl)
        print('Success!')
    elif args.data_cache:
        print('Starting data cache system')
        setup(do_model=False)
        data_saver(args.epoch_count)
        print('Done!')
    else:
        setup(do_data=not args.optimized)
        main(epoch_count=args.epoch_count, stop_limit=args.stop_num, save=args.save)
