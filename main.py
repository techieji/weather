import nn
import torch

def main():
    nn.load_all()
    _sl = nn.random_slice()
    sl = nn.to_tensors(_sl)
    print(sl.shape)
    regenc = nn.RegionEncoder()
    lo = nn.LiquidOperator()
    ts = regenc(sl)
    r = lo(ts)
    return r

if __name__ == '__main__':
    main()
