# model managment functions
# last edited 191126 - tidying

import torch
import torch.nn as nn
from src.architectures import tennet, sevennet, eightnnet, sixnet, smallnet1, smallnet2, smallnet3
import pickle

#  SELECT AND INIT MODEL VIA  MODEL NAME (STR) #  SELECT AND INIT MODEL VIA  MODEL NAME (STR)
def choose_model(model_name, lin_lay, dropout, output_lin_lay=11):
    # Takes in model name, firt linear layer value, dropout value and output value
    # Returns specified model
    if model_name == '4c3l':
        return smallnet1(in_chan=3, f_lin_lay=int(lin_lay), l_lin_lay=output_lin_lay, ks= (3,5), dropout= dropout)
    elif model_name == '3c2l':
        return smallnet2(in_chan=3, f_lin_lay=int(lin_lay), l_lin_lay=output_lin_lay, ks = (3,5), dropout=dropout)
    elif model_name == '2c2l':
        return smallnet3(in_chan=3, f_lin_lay=int(lin_lay), l_lin_lay=output_lin_lay, ks= (3,5), dropout= dropout)
    if model_name == '6c3l':
        return sixnet(in_chan=3, f_lin_lay=int(lin_lay), l_lin_lay=output_lin_lay, ks= (3,5), dropout= dropout)
    elif model_name == '7c3l':
        return sevennet(in_chan=3, f_lin_lay=int(lin_lay), l_lin_lay=output_lin_lay, ks= (3,5), dropout= dropout)
    elif model_name == '8c3l':
        return eightnnet(in_chan=3, f_lin_lay=int(lin_lay), l_lin_lay=output_lin_lay, ks= (3,5), dropout= dropout)
    elif model_name == '10c4l':
        return tennet(in_chan=3, f_lin_lay=int(lin_lay), l_lin_lay = output_lin_lay, ks=(3, 5), dropout=dropout)
    elif model_name in ['vgg16', 'vgg', 'VGG', 'VGG16']:
        from torchvision.models import vgg16
        model_vgg16 = vgg16()
        vgg_classifier = model_vgg16.classifier # "By default, no pre-trained weights are used."
        vgg_classifier.pop(6)
        vgg_mod = nn.Sequential(
            model_vgg16.features,
            nn.Flatten(),
            vgg_classifier,
            nn.Linear(4096,output_lin_lay), # cheanging the output layer
            nn.Softmax(dim=0),  
            )
                
        return vgg_mod
    else:
        print('Model Name Not Recognised')


def load_pretrained_model(dir_pkl, pkl_name, model_name, res:str):
    # Takes in directory for pickle file, pickle file name, model name, and resolution
    # returns model loaded with pretrained weights
    # useful for further training
    from modelCards import Cards
    C = Cards() 
    
    linlay = C.modname2linlay(model_name, res)
    model = choose_model(model_name, linlay, dropout=0.2) #
    print("Loading Weights into model...")
    try:
        with open(dir_pkl+pkl_name, 'rb') as f:
            model_pkl = torch.load(f)
            print("torch load")
    except:
        with open(dir_pkl+pkl_name, 'rb') as f:
            model_pkl = pickle.load(f)
            print("pkl load")
            
    model.load_state_dict(model_pkl['model.state_dict'])
    print("Done")
    return model


def choose_scheduler(save_dict, optimizer):
    # Handles scheduler managment
    # takes in save_dict : a dictionary of simulation parameters, 
    # takes in optimizer : string of
    #  "RoP" -reduce on plateau
    #  "Exp" -Exponential
    # "NoSched" Returns None
    # Returns specified torch scheduler
    import torch.optim.lr_scheduler as lr_scheduler
    if save_dict['scheduler'] == "RoP":
        scheduler = lr_scheduler.ReduceLROnPlateau(optimizer, mode='min')#, factor=0.1, patience=10, threshold=10)
    elif save_dict['scheduler'] == "Exp":
        scheduler = lr_scheduler.ExponentialLR(optimizer, gamma=0.95, last_epoch=-1, verbose='deprecated') 
    elif save_dict['scheduler'] == "NoSched":
        scheduler = None
    return scheduler

def get_seeds(resolution, modelname, picklePath):
    # Handles retriving the random seed value of a saved model
    # useful for further training
    # seed value is contained within file name
    # position changes depending on resolution (appears before seed in file name)
	import os
	pkl_seeds = []
	pklStart = 72
	pklEnd = 75
	pkl10Start = 71
	pkl10End = 74

	minusone = [[226, 72],[113, 36]]
	minustwo =[[57, 18]]
	minusthree = [[29, 9],[15, 5]]
	minusfour = [[8, 3]]

	if resolution in minusone:
		pklStart = pklStart -3 # 3*1
		pklEnd = pklEnd -3
		pkl10Start = pkl10Start -3
		pkl10End = pkl10End -3
	elif resolution in minustwo:
		pklStart = pklStart -6 # 3*2
		pklEnd = pklEnd -6
		pkl10Start = pkl10Start -6
		pkl10End = pkl10End -6
	elif resolution in minusthree:
		pklstart = pklstart - 9 # 3*3
		pklEnd = pklEnd -9
		pkl10Start = pkl10Start -9
		pkl10End = pkl10End -9
	elif resolution in minusfour:
		pklStart = pklStart -12 # 3*4
		pklEnd = pklEnd -12
		pkl10Start = pkl10Start -12
		pkl10End =pkl10End - 12

	for file in os.listdir(picklePath):
		if file[-3:] == "pkl":
			if modelname != "10c4l":
				print(file[pklStart:pklEnd].replace("_",""))
				if file[pklStart:pklEnd].replace("_","").isdecimal():
					pkl_seeds.append(int(file[pklStart:pklEnd].replace("_","")))
					print(f"File Check : {file[pklStart:pklEnd].replace('_','')}")
			else:
				print(file[pkl10Start:pkl10End].replace("_",""))
				if file[pkl10Start:pkl10End].replace("_","").isdecimal():
					pkl_seeds.append(int(file[pkl10Start:pkl10End].replace("_","")))
	return pkl_seeds
