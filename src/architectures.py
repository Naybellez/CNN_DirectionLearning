## created 120923
# last edited 191126 - tidying

# Script containing model architectures used in Direction Learning study

# Imports
import torch.nn as nn

def tennet(in_chan, f_lin_lay, l_lin_lay, ks, dropout): #  10 conv layers 4 linear
    class TenNet(nn.Module):
        def __init__(self):
            super(TenNet, self).__init__()
            self.flatten = nn.Flatten()
            self.conv_layers = nn.Sequential(
                nn.Conv2d(in_channels=in_chan, out_channels=32, kernel_size=ks, padding=2),
                nn.ReLU(),
                nn.Dropout(p=dropout),
                nn.Conv2d(in_channels=32, out_channels=32,kernel_size=ks),
                nn.ReLU(),
                nn.Conv2d(in_channels=32, out_channels=64, kernel_size = ks, padding=2),
                nn.ReLU(),
                nn.Conv2d(in_channels=64, out_channels=64, kernel_size=ks),
                nn.ReLU(),
                nn.MaxPool2d(2, 2),
                nn.Conv2d(in_channels=64, out_channels=128, kernel_size=ks, padding=2),
                nn.ReLU(),
                nn.Conv2d(in_channels=128, out_channels=128, kernel_size=ks),
                nn.ReLU(),
                nn.MaxPool2d(2, 2),
                nn.Conv2d(in_channels=128, out_channels=256, kernel_size=ks, padding=2),
                nn.ReLU(),
                nn.Conv2d(in_channels=256, out_channels=256, kernel_size=ks),
                nn.ReLU(),
                nn.MaxPool2d(2,2),
                nn.Conv2d(in_channels=256, out_channels=512, kernel_size=ks, padding=2),
                nn.ReLU(),
                nn.Conv2d(in_channels=512, out_channels=512, kernel_size=ks),
                nn.ReLU(),
                nn.MaxPool2d(2,2),
                nn.Dropout(p=dropout)
               )
            self.linear_1 = nn.Sequential(
                nn.Linear(f_lin_lay, 100),
                nn.ReLU(),
                nn.Linear(100, 100),
                nn.ReLU(),
                nn.Linear(100,200),
                nn.ReLU(),
                nn.Linear(100, l_lin_lay),
                nn.Softmax()
               )
        def forward(self, x):
           x = self.conv_layers(x)
           x = x.flatten(start_dim=1)
           x = x.squeeze()
           x = self.linear_1(x)
           return x

    model = TenNet()
    return model 

def eightnnet(in_chan, f_lin_lay, l_lin_lay, ks, dropout): # 8 conv layers
    class Eightnet(nn.Module):
        # 8 conv layers, 3 linear
        def __init__(self):
            super(Eightnet, self).__init__()
            self.flatten = nn.Flatten()
            self.conv_layers = nn.Sequential( 
                  nn.Conv2d(in_channels=in_chan, out_channels=32, kernel_size=ks, padding=2),
                  nn.ReLU(), 
                  nn.Dropout(p=dropout),
                 nn.Conv2d(in_channels=32, out_channels=32, kernel_size=ks, padding=2),
                  nn.ReLU(), 
                  nn.Conv2d(in_channels=32, out_channels=64, kernel_size=ks, padding=2),
                  nn.ReLU(),
                  nn.Conv2d(in_channels =64, out_channels=64, kernel_size=ks),
                  nn.ReLU(),
                  nn.MaxPool2d(2, 2),
                  nn.Conv2d(in_channels=64, out_channels=128, kernel_size=ks, padding=2),
                  nn.ReLU(), 
                  nn.Conv2d(in_channels =128, out_channels=128, kernel_size=ks),
                  nn.ReLU(),
                  nn.MaxPool2d(2,2),
                  nn.Conv2d(in_channels=128, out_channels=256, kernel_size=ks, padding=2),
                  nn.ReLU(), 
                  nn.Conv2d(in_channels =256, out_channels=256, kernel_size=ks),
                  nn.ReLU(),
                  nn.MaxPool2d(2,2),
                  nn.Dropout(p=dropout),
              )

            self.linear_1 = nn.Sequential( 
                nn.Linear(f_lin_lay, 100),
                nn.ReLU(),
                nn.Linear(100,100),
                nn.ReLU(),
                nn.Dropout(p=dropout),
                nn.Linear(100,l_lin_lay),
                nn.Softmax(),
            )

        def forward(self, x):
          #forward method. opposition to backward pass
          x= self.conv_layers(x)
          x = x.flatten(start_dim=1)
          x = x.squeeze()
          x = self.linear_1(x)
          return x
    model = Eightnet()
    return model

def sevennet(in_chan, f_lin_lay, l_lin_lay, ks, dropout): # 7 conv layers 
    class SevenNet(nn.Module):
        # 7 conv layers, 3 linear
        def __init__(self):
            super(SevenNet, self).__init__()
            self.flatten = nn.Flatten()
            self.conv_layers = nn.Sequential(  
                  nn.Conv2d(in_channels=in_chan, out_channels=32, kernel_size=ks, padding=2),
                  nn.ReLU(), 
                  nn.Dropout(p=dropout),
                  nn.Conv2d(in_channels=32, out_channels=64, kernel_size=ks, padding=2),
                  nn.ReLU(), 
                  nn.Conv2d(in_channels =64, out_channels=64, kernel_size=ks),
                  nn.ReLU(),
                  nn.MaxPool2d(2, 2),
                  nn.Conv2d(in_channels=64, out_channels=128, kernel_size=ks, padding=2),
                  nn.ReLU(), 
                  nn.Conv2d(in_channels =128, out_channels=128, kernel_size=ks),
                  nn.ReLU(),
                  nn.MaxPool2d(2,2),
                  nn.Conv2d(in_channels=128, out_channels=256, kernel_size=ks, padding=2),
                  nn.ReLU(), 
                  nn.Conv2d(in_channels =256, out_channels=256, kernel_size=ks),
                  nn.ReLU(),
                  nn.MaxPool2d(2,2),
                  nn.Dropout(p=dropout), 
              )

            self.linear_1 = nn.Sequential(  
                nn.Linear(f_lin_lay, 100),
                nn.ReLU(),
                nn.Linear(100,100),
                nn.ReLU(),
                nn.Dropout(p=dropout),
                nn.Linear(100,l_lin_lay),
                nn.Softmax(),
            )

        def forward(self, x):
          #forward method. opposition to backward pass
          x= self.conv_layers(x)
          x = x.flatten(start_dim=1)#start_dim=1
          x = x.squeeze()
          x = self.linear_1(x)
          return x
    model = SevenNet()
    return model

def sixnet(in_chan, f_lin_lay, l_lin_lay, ks, dropout): # 6 conv layers
    class Sixnet(nn.Module):
        # 6 conv layers, 3 linear
        def __init__(self):
            super(Sixnet, self).__init__()
            self.flatten = nn.Flatten()
            self.conv_layers = nn.Sequential(  
                  nn.Conv2d(in_channels=in_chan, out_channels=64, kernel_size=ks, padding=2),
                  nn.ReLU(), 
                  nn.Dropout(p=dropout),
                  nn.Conv2d(in_channels=64, out_channels=64, kernel_size=ks, padding=2),
                  nn.ReLU(), 
                  nn.Conv2d(in_channels =64, out_channels=128, kernel_size=ks),
                  nn.ReLU(),
                  nn.MaxPool2d(2, 2),
                  nn.Conv2d(in_channels=128, out_channels=128, kernel_size=ks, padding=2),
                  nn.ReLU(), 
                  nn.Conv2d(in_channels =128, out_channels=256, kernel_size=ks),
                  nn.ReLU(),
                  nn.MaxPool2d(2,2),
                  nn.Conv2d(in_channels=256, out_channels=256, kernel_size=ks, padding=2),
                  nn.ReLU(),
                  nn.MaxPool2d(2,2),
                  nn.Dropout(p=dropout), 
              )

            self.linear_1 = nn.Sequential( 
                nn.Linear(f_lin_lay, 100),
                nn.ReLU(),
                nn.Linear(100,100),
                nn.ReLU(),
                nn.Dropout(p=dropout),
                nn.Linear(100,l_lin_lay),
                nn.Softmax(),
            )

        def forward(self, x):
          # forward method. opposition to backward pass
          x= self.conv_layers(x)
          x = x.flatten(start_dim=1)
          x = x.squeeze()
          x = self.linear_1(x)
          return x
    model = Sixnet()
    return model


def smallnet1(in_chan, f_lin_lay, l_lin_lay, ks, dropout=0.5):  # 4 conv layers
    class SmallNet1(nn.Module):
        # 4 conv layers, 3 linear
        def __init__(self):
            super(SmallNet1, self).__init__()
            self.flatten = nn.Flatten()
            self.conv_layers = nn.Sequential( 
                  nn.Conv2d(in_channels=in_chan, out_channels=32, kernel_size=ks, padding=2),
                  nn.ReLU(),
                  nn.Dropout(p=dropout),
                  nn.Conv2d(in_channels=32, out_channels=64, kernel_size=ks, padding=2),
                  nn.ReLU(), 
                  nn.Conv2d(in_channels =64, out_channels=64, kernel_size=ks),
                  nn.ReLU(),
                  nn.MaxPool2d(2, 2),
                  nn.Conv2d(in_channels=64, out_channels=128, kernel_size=ks, padding=2),
                  nn.ReLU(),
                  nn.MaxPool2d(2,2),
                  nn.Dropout(p=dropout),
              )
            self.linear_1 = nn.Sequential( 
                nn.Linear(f_lin_lay, 100),
                nn.ReLU(),
                nn.Linear(100,100),
                nn.ReLU(),
                nn.Dropout(p=dropout),
                nn.Linear(100,l_lin_lay),
                nn.Softmax(),
            )

        def forward(self, x):
          x= self.conv_layers(x)
          x = x.flatten(start_dim=1)
          x = x.squeeze()
          x = self.linear_1(x)
          return x
    model = SmallNet1()
    return model



def smallnet2(in_chan, f_lin_lay, l_lin_lay, ks, dropout=0.5, start_dim=1): # 3 conv layers
    class SmallNet2(nn.Module):
        # 3 conv layers, 2 linear
        def __init__(self):
            super(SmallNet2, self).__init__()
            self.flatten = nn.Flatten()
            self.conv_layers = nn.Sequential( 
                  nn.Conv2d(in_channels=in_chan, out_channels=32, kernel_size=ks, padding=2),
                  nn.ReLU(), 
                  nn.Dropout(p=dropout),
                  nn.Conv2d(in_channels=32, out_channels=64, kernel_size=ks, padding=2),
                  nn.ReLU(),
                  nn.Conv2d(in_channels =64, out_channels=64, kernel_size=ks),
                  nn.ReLU(),
                  nn.MaxPool2d(2,2),
                  nn.Dropout(p=dropout),
              )
            self.linear_1 = nn.Sequential(
                nn.Linear(f_lin_lay, 100),
                nn.ReLU(),
                nn.Dropout(p=dropout),
                nn.Linear(100,l_lin_lay),
                nn.Softmax(),
            )

        def forward(self, x):
          x = self.conv_layers(x)
          x = x.flatten(start_dim=start_dim)
          x = x.squeeze()
          x = self.linear_1(x)
          return x
    model = SmallNet2()
    return model


def smallnet3(in_chan, f_lin_lay, l_lin_lay, ks, dropout=0.5, start_dim=1):  # 2 conv layers
    class SmallNet3(nn.Module):
        # 2 conv layers, 2 linear
        def __init__(self):
            super(SmallNet3, self).__init__()
            self.flatten = nn.Flatten()
            self.conv_layers = nn.Sequential( 
                  nn.Conv2d(in_channels=in_chan, out_channels=32, kernel_size=ks, padding=2),
                  nn.ReLU(), #inplace=True
                  nn.Dropout(p=dropout),
                  nn.Conv2d(in_channels =32, out_channels=64, kernel_size=ks),
                  nn.ReLU(),
                  nn.MaxPool2d(2,2),
                  nn.Dropout(p=dropout),
              )

            self.linear_1 = nn.Sequential(
                nn.Linear(f_lin_lay, 100),
                nn.ReLU(),
                nn.Dropout(p=dropout),
                nn.Linear(100,l_lin_lay),
                nn.Softmax(),
            )

        def forward(self, x):
          x= self.conv_layers(x)
          x = x.flatten(start_dim=start_dim)
          x = x.squeeze()
          x = self.linear_1(x)
          return x
    model = SmallNet3()
    return model


# usefull classes
class PrintLayer(nn.Module):
	# useful for printing network layers
	def __init__(self):
		super(PrintLayer, self).__init__()
	def forward(self, x):
		print(x)
		return x

class Squeeze(nn.Module):
	# Handles squeezing tensors
	def __init__(self):
		super(Squeeze, self).__init__()
	def forward(self, x):
		x = x.squeeze(0)
		return x

class Flattern(nn.Module):
    # Handles flattening tensors
    def __init__(self):
        super(Flattern, self).__init__()
    def forward(self, x):
        x = x.flatten()
        return x
