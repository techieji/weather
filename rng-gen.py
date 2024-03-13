import nn
from main import *
#import main
import sys

model = ModelData.load('scewhiq_xocug').model

print('Start', file=sys.stderr)

for i, sl in enumerate(nn.random_slice_iter()):
    pos = nn.random_pos()
    loss = calc_loss(sl, model=model, needs_grad=False, train_mode=False)
    print(pos, loss)
    print(i, end='\r', file=sys.stderr)
