# script containing function that handles retreiving loss function
# last edited 191126 - tidying
import torch.nn as nn

def set_lossfn(lf):
    # takes in lf: string specifying loss function to use
    # returns torch.nn loss function
    print(f"set_lossfn    lf  {lf}")
    if isinstance(lf, list):
        lf = lf[0]
    if lf =='MSE':
        loss_fn = nn.MSELoss()
    elif lf == 'CrossEntropy':
        loss_fn = nn.CrossEntropyLoss()
    else:
        loss_fn = lf
    return loss_fn
