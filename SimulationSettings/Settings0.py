import os
from src.modelCardsP3Direction import Cards, return_card
import random

# This is the settings retrived if GPU 0 is chosen
# Allows for parallelising simualations

save_dir  = "saves/"
data_path = "../../" # Replace with path to data folder
gitHASH = ""


model_name = '3c2l' # Replace with the model you'd like to run
# '2c2l', '3c2l', '4c3l', 6c3l', '7c3l', '8c3l' 
epochs = 300
tv = "R3" # Replace with subproject code
learning_rate = 1e-4

# Get list pf random seed values
T = 0
total_seeds = 2 # replace with number of repeat simulations (number of random seeds)
seeds = []
while len(seeds) <= total_seeds:
    seeds.append(random.randint(0, 1000))
    T +=1

batchsize = 64 # 32 for 6c3l (memory constraints)
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


# Retrive model cards
cards = Cards()
modelcards = cards.modelcards
if model_name!= 'resnet18':
    modelcard = return_card(modelcards, key='name', targetValue=model_name)[0]
    print('Model Card:  ',modelcard)

# Retrive resolution cards and add to list for looping
resolutioncards = cards.resolutioncards
rescard_452 = return_card(resolutioncards, key ='resolution', targetValue=[452, 144])[0]
rescard_226 = return_card(resolutioncards, key ='resolution', targetValue=[226, 72])[0]
rescard_113 = return_card(resolutioncards, key ='resolution', targetValue=[113, 36])[0]
rescard_57 = return_card(resolutioncards, key ='resolution', targetValue=[57, 18])[0]
rescard_29 = return_card(resolutioncards, key ='resolution', targetValue=[29, 9])[0]
rescard_15 = return_card(resolutioncards, key ='resolution', targetValue=[15, 5])[0]
rescard_8 = return_card(resolutioncards, key ='resolution', targetValue=[8, 3])[0]

all_rescards = [rescard_452, rescard_226, rescard_113, rescard_57, rescard_29] 

