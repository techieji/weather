import nn
import torch
import numpy as np

nn.load_all()

# Old model, useful for loading old versions
# _model = torch.nn.Sequential(
#     nn.RegionEncoder(),
#     nn.LiquidOperator()
# )

model = nn.WeatherPredictor()

LR = 0.1

optim = torch.optim.Adam(model.parameters(), lr=LR)
loss = torch.nn.MSELoss()
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optim)

def calc_loss(sl=None, model=model, needs_grad=True, train_mode=True, n=1):
    try:
        model.train(train_mode)
    except:
        pass
    ret = None
    for _ in range(n):
        if sl is None:
            _sl = nn.random_slice()
            sl = nn.to_tensors(_sl)
        warmup = sl[:nn.TIME_LENGTH]
        warmup.requires_grad_(needs_grad)
        r = model(warmup)
        value = sl[nn.TIME_LENGTH:,:,nn.GLOBAL_TILE,nn.GLOBAL_TILE,:].flatten(1)
        rvalue = r[nn.TIME_LENGTH:]
        # print(value.shape, rvalue.shape)
        if ret is None:
            ret = loss(value, rvalue)
        else:
            ret += loss(value, rvalue)
    return ret/n

def train(n=1):
    l = calc_loss(n=n)
    optim.zero_grad()
    l.backward()
    optim.step()
    return l.sum().item()

class Validator:
    def __init__(self, n=10):
        print('Setting up validator...')
        self.n = n
        self.slices = []
        self._const_sels = []
        for _ in range(n):
            self.slices.append(nn.to_tensors(nn.random_slice()))
            self._const_sels.append(nn._const_sel)
        print('Validator is set up')

    def _switch_to(self, n):
        nn._const_sel = self._const_sels[n]
        return self.slices[n]

    def __call__(self):
        with torch.no_grad():
            losses = []
            for i in range(self.n):
                sl = self._switch_to(i)
                losses.append(calc_loss(sl,train_mode=False))
            return sum(losses)/len(losses)

def advance_world(world, t, model=model):
    pass

validate = Validator()

def save_onnx():
    if not nn.is_data_loaded():
        nn.load_all()
    sl = nn.to_tensors(nn.random_slice())[:nn.TIME_LENGTH]
    export = torch.onnx.dynamo_export(lambda a, b: model(a), sl, nn._const_sel)
    export.save('model1.onnx')

def main(epoch_count=100):
    if not nn.is_data_loaded():
        nn.load_all()
    last_val_loss = np.inf
    stop_n = 0
    for i in range(epoch_count):
        if stop_n > 5:
            print('Training stopped')
            break
        try:
            train_loss = train(n=1)
            val_loss = validate().sum()
            yield (val_loss.item(), train_loss)
            if val_loss - last_val_loss > 0:
                stop_n += 1
            else:
                stop_n = 0
            last_val_loss = val_loss
            scheduler.step(val_loss)
        except Exception as e:
            print(f'Error ({i}):', e)

if __name__ == '__main__':
    pass
