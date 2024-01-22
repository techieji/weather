import nn
import torch
import torch.nn.functional as F
import numpy as np
from time import time
import json

import sys
import argparse

model = None
validate = None

# def loss(value, mvalue, anom):
#     print('nans:', value.isnan().sum(), mvalue.isnan().sum(), anom.isnan().sum())
#     pcc = nn.anomaly_correlation(value, mvalue, anom)
#     return F.mse_loss(mvalue, value) * -pcc

loss = torch.nn.L1Loss()

def setup():
    global model, validate, optim, scheduler
    nn.load_all()
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
        print('Setting up validator...')
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

def _main(epoch_count=50, stop_limit=5):
    if not nn.is_data_loaded():
        nn.load_all()
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

def main(*args, **kwargs):
    print('Calculating base loss')
    base_loss, base_pcc = validate(model=nn.persistence_model)
    base_loss = base_loss.item()
    base_pcc = base_pcc.item()
    print('Base loss:', base_loss)
    print('Base PCC :', base_pcc)
    print('Epoch\tLoss\tDiff\tPCC')
    print('-----\t----\t----\t---')
    lossl = []
    for i, loss in enumerate(_main(*args, **kwargs)):
        lossl.append(loss)
        print(i, loss[0], base_loss - loss[0], loss[2], sep='\t')
    print('Final diff:', base_loss - loss[0])
    # if base_loss - loss[0] > 0:
    if True:
        print('success!')
        torch.save(model.state_dict(), f'model-last-{time()}.pt')
        with open(f'model-loss-last-{time()}.json', 'w') as f:
            json.dump(lossl, f)
    else:
        print('fail.')

if __name__ == '__main__':
    parser = argparse.ArgumentParser(prog='weather-trainer')
    parser.add_argument('-f', '--file')
    parser.add_argument('-e', '--epoch-count', type=int, default=100)
    parser.add_argument('-s', '--stop-num', type=int, default=5)
    parser.add_argument('-t', '--test-forwards', action='store_true')
    args = parser.parse_args()
    if args.file is not None:
        state_dict = torch.load(args.file)
        model.load_state_dict(state_dict)
    if args.test_forwards:
        diam = 2*nn.GLOBAL_TILE - 1
        nn._const_sel = torch.zeros((diam, diam, nn.VAR_C))
        sl = torch.zeros((nn.TIME_LENGTH, nn.VAR_N, diam, diam, nn.LEVELS))
        print('Starting test')
        model = nn.WeatherPredictor()
        model(sl)
        print('Success!')
    else:
        setup()
        main(epoch_count=args.epoch_count, stop_limit=args.stop_num)
