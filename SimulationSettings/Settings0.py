import os
from src.modelCardsP3Direction import Cards, return_card
import random


save_dir  = "saves/"
data_path = "/its/home/nn268/antvis/antvis/optics/NC_IDSW/"
gitHASH = " 32d139c134ef0530871dc082bae2923877f71936"


model_name = '3c2l'
epochs = 300
#tv = [226, 72]
tv = "R3"
learning_rate = 1e-4

T = 0
seeds = []
while len(seeds) <= 2:
    seeds.append(random.randint(0, 1000))#[42, 7, 56, 23, 22, 69, 100]
    T +=1

batchsize = 64 # 32
half_ciprange = 22 # (roughly half of 45)
std_dev = 7

output_lin_lay = 360 ###### Output labels for direction prediction specifically. 

loss_fn = ['MSE']
optim = ["adam"]
scheduler_value = "NoSched"

projectNAME =f"R3_{model_name}_300E_1e-4_ADAM_{tv}"

full_path = save_dir+"R3/"+f"{tv}"+"/"+model_name+"/"+projectNAME
if not os.path.exists(full_path):
    os.makedirs(full_path)
save_location = full_path


cards = Cards()
modelcards = cards.modelcards
if model_name!= 'resnet18':
    modelcard = return_card(modelcards, key='name', targetValue=model_name)[0]
    print('Model Card:  ',modelcard)

resolutioncards = cards.resolutioncards
rescard_452 = return_card(resolutioncards, key ='resolution', targetValue=[452, 144])[0]
rescard_226 = return_card(resolutioncards, key ='resolution', targetValue=[226, 72])[0]
rescard_113 = return_card(resolutioncards, key ='resolution', targetValue=[113, 36])[0]
rescard_57 = return_card(resolutioncards, key ='resolution', targetValue=[57, 18])[0]
rescard_29 = return_card(resolutioncards, key ='resolution', targetValue=[29, 9])[0]
rescard_15 = return_card(resolutioncards, key ='resolution', targetValue=[15, 5])[0]
rescard_8 = return_card(resolutioncards, key ='resolution', targetValue=[8, 3])[0]

all_rescards = [rescard_452, rescard_226, rescard_113, rescard_57, rescard_29] # rescard_452, rescard_226,rescard_113,, rescard_15, rescard_8]

