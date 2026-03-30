import os
import torch
from src.modelCardsP3Direction import Cards, return_card
import random

save_dir  = "saves/"
data_path = "/its/home/nn268/antvis/antvis/optics/NC_IDSW/"

gitHASH = " 32d139c134ef0530871dc082bae2923877f71936"

model_name = '7c3l'
epochs = 300
tv = [8, 3]
learning_rate = 1e-4

T = 0
seeds = []
while len(seeds) <= 7:
    seeds.append(random.randint(1, 500))#[42, 7, 56, 23, 22, 69, 100]
    T +=1
    
batchsize = 64
half_ciprange = 22 # (roughly half of 45)
std_dev = 7

output_lin_lay = 360 ###### Output labels for direction prediction specifically. 

loss_fn = ['MSE']
optim = ["adam"]
scheduler_value = "NoSched"

projectNAME = f"{model_name}_300E_1e-4_ADAM_{tv}"

full_path = save_dir+f"{tv}"+"/"+model_name+"/"+projectNAME
if not os.path.exists(full_path):
    os.makedirs(full_path)
save_location = full_path


cards = Cards()
modelcards = cards.modelcards
if model_name!= 'resnet18':
    modelcard = return_card(modelcards, key='name', targetValue=model_name)[0]
    print(modelcard)
    
resolutioncards = cards.resolutioncards
resolutioncard = return_card(resolutioncards, key ='resolution', targetValue=tv)
