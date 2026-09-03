from src.GradCam_class import CAMfromPickle
from src.modelManagment import get_seeds, choose_model
from src.modelCardsP3Direction import Cards, return_card, get_lin_lay
import random
import torch

res = [29, 9] # 57, 18
model_name = "2c2l"

picklePath = f"/its/home/nn268/antvis/antvis/CNN_DirectionLearning/saves/{res}/{model_name}/"


subfolder = 'testingAll/clean/'
parentDir = f"/its/home/nn268/antvis/antvis/CNN_DirectionLearning/saves/tests/CAM/"

epochs = 300
lr = "1e-4"


pkl_seeds = get_seeds(resolution=res, modelname=model_name, picklePath=picklePath)
#index = random.randint(0, len(pkl_seeds)-1)
#seed = pkl_seeds[index]
for seed in pkl_seeds:

	if model_name != "10c4l":
		pickle_file = f"{model_name}_{epochs}E_{lr}_ADAM_{res}{model_name}_{res}_{res}_0.0001_NoSched_{seed}_MSE.pkl"
	else:
		pickle_file = f"TESTING_{model_name}_{epochs}_{res}{model_name}_{res}_{res}_0.0001_NoSched_{seed}_MSE.pkl"


	CAM = CAMfromPickle(model_file=pickle_file, dir=picklePath, data_path="/its/home/nn268/antvis/antvis/optics/NC_IDSW/", modelname=model_name, seed=seed, resolution=res, subfolder=subfolder, parentDir=parentDir, device='cpu')

	cards = Cards()
	modelcards = cards.modelcards
	if model_name != "resnet18":
		modelcard = return_card(modelcards, key='name', targetValue = model_name)[0]
		print(f"Model Cards : {modelcard}")

	resolutioncards = cards.resolutioncards
	resolutioncard = return_card(resolutioncards, key ='resolution', targetValue = res)

	linlay = get_lin_lay(modelcard, res)
	model = choose_model(model_name, linlay, 0, 360).to('cpu')
	torch.cuda.empty_cache()


	CAM.get_test_set()
	CAM.fill_model(model)

	CAM.createCAMFig()
