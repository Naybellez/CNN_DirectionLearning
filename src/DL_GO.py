# last edited 191126
# Script contains run_go and _go, the simulation functions
import wandb
import time 
import numpy as np
import torch
from torch.utils.data import DataLoader
from torchvision.models import resnet18

from src.modelCardsP3Direction import get_lin_lay
from src.functions import  ImageProcessor
from src.modelManagment import choose_model
from src.dataPreProcessingP3Direction import get_data
from src.dataloaderP3Direction import IDSWDataSetLoader3
from src.SimManagment import set_lossfn
from src.loopsP3Direction import test_loop_batch, train_val_batch

def run_go(GPU):
    # Handles GPU init and _go function
    if GPU == 0:
        device = "cuda:0" if torch.cuda.is_available() else "cpu"
        import SimulationSettings.Settings0 as SS 
        torch.cuda.empty_cache()

    elif GPU == 1:
        device = "cuda:1" if torch.cuda.is_available() else "cpu"
        import SimulationSettings.Settings1 as SS
        torch.cuda.empty_cache()

    print(f"run_go:  {GPU}  {device}")
    torch.cuda.empty_cache()

    def _go(config=None):
        # Handles model training and testing
        # Handles looping though resolution values
        # One run results in multiple simulation with different parameters
        if len(SS.gitHASH) <1:
            print("YOU FORGET THE GIT HASH")
            return
        else:
            print('Git Hash registered')
        
        with wandb.init(mode="offline", config=config):  
            config = wandb.config

            model_card = SS.modelcard
            print(model_card)
            model_name = model_card['model']
            dropout = model_card['dropout'] 

            for res_idx, resolution_card in enumerate(SS.all_rescards):
                resolution = resolution_card['resolution']
                lin_lay = get_lin_lay(model_card, resolution) # returns the value of the expected size for input to fully connected layer
                seedNum = np.random.randint(len(SS.seeds))
                seed = SS.seeds[seedNum]
                loss_type = 'MSE'
                loss_fn = set_lossfn(loss_type)
                batch = SS.batchsize

                # SIMULATION SET OFF PRINTS # SIMULATION SET OFF PRINTS # SIMULATION SET OFF PRINTS # SIMULATION SET OFF PRINTS 
                print('Model: ', str(model_name), f" {len(SS.modelcards)}")
                print('resolution: ', str(resolution), f" idx: {res_idx} / {len(SS.all_rescards)}")
                print('seed: ', str(seed))
                print('loss function: ', str(loss_type))
                print('Batch size: ', SS.batchsize)
                print('Training epochs: ', SS.epochs)
                print(device)
                run_start_time = time.process_time()
                print('start time: ',run_start_time)

                epochs = SS.epochs #40

                IP = ImageProcessor(device)
                # unhash if using wandb simulation manager
                #wandb.log({'gitHash':SS.gitHASH})
                        #wandb.log({'Epochs': epochs})
                        #wandb.log({'schedType':SS.scheduler_value})

                        # DICTIONARY TO HOLD SIMULATION SETTINGS
                save_dict = {'Run' : f"{model_name}_{resolution}",
                        'start_epoch' : 0,
                        'Current_Epoch': 0,
                        'save_location' : SS.save_location,
                        'scheduler': SS.scheduler_value,
                        'gitHASH':str(SS.gitHASH),
                        'model_name': str(model_name),
                        'loss_fn': str(loss_type),
                        'lr': str(SS.learning_rate),
                        'resolution': str(resolution),
                        'seed': str(seed),
                        'lin_lay': int(lin_lay)}


                # SELECTING THE MODEL BASED ON MODEL_NAME
                if model_name == 'resnet18':
                    model = resnet18(weights=None, num_classes =360).to(device)
                else:
                    model = choose_model(model_name, lin_lay, dropout, SS.output_lin_lay).to(device)
                torch.cuda.empty_cache()

                # DATALOADING # DATALOADING # DATALOADING # DATALOADING # DATALOADING # DATALOADING # DATALOADING # DATALOADING 
                print("Model Loaded.  \n Loading Data...")
                x_train, _, x_val, _, x_test, y_test = get_data(seed, SS.data_path)
                print("LEN x train:  ",len(x_train))
                av_lum = IP.new_luminance(x_train)
                train = (x_train, resolution, av_lum, model_name, SS.half_ciprange, SS.std_dev, batch)

                test_ds = IDSWDataSetLoader3(x_test, resolution,av_lum,model_name, SS.half_ciprange, SS.std_dev, device)
                print("LEN test custom loader : ", len(test_ds))
                test = DataLoader(test_ds, batch_size=SS.batchsize, shuffle=True, drop_last=True) #, num_workers=2
                print("LEN test loader : ",len(test))
                # set optimizer
                optimizer = torch.optim.Adam(model.parameters(),lr=SS.learning_rate)

                # Tell simualtion manager (WANDB) how frequently to log progress
                #wandb.watch(model, loss_fn, log='all', log_freq=2, idx = model_index)

                # CREATE UNIQUE SIMULATION RUN NAME 
                loop_run_name = f"{save_dict['Run']}_{resolution}_{SS.learning_rate}_{SS.scheduler_value}_{seed}_{loss_type}"

                # TRAINING # TRAINING # TRAINING # TRAINING # TRAINING # TRAINING # TRAINING # TRAINING # TRAINING # TRAINING # TRAINING # TRAINING 
                model, save_dict = train_val_batch(model, train, x_val, save_dict,  loss_fn, epochs, optimizer, device, config, SS)

                # TESTING # TESTING # TESTING # TESTING # TESTING # TESTING # TESTING # TESTING # TESTING # TESTING # TESTING # TESTING 
                test_acc, test_predict_list, y_test = test_loop_batch(model,test, device, config, SS, runname=loop_run_name, save_loc=save_dict['save_location']) 

                # Prep data for plotting
                #test_predict_numerical = [p.detach().to('cpu').numpy() for p in test_predict_list] #.detach().to('cpu').numpy()
                #y_test_numerical = [y.detach().to('cpu').numpy() for y in y_test]

                # PRINTING AND SAVING # PRINTING AND SAVING # PRINTING AND SAVING # PRINTING AND SAVING # PRINTING AND SAVING # PRINTING AND SAVING 
                #wandb.log({'test_predict': test_predict_list})
                #wandb.log({'test_labels': list(y_test)})
                save_dict['TESTAccBase'] = test_acc['BaseAcc']
                save_dict['TESTAccMSE'] = test_acc['MSE']
                save_dict['TESTAccMAE'] = test_acc['MAE']
                save_dict['TESTAccPeakDist'] = test_acc['peakDist']
                save_dict.update({'test_predict': test_predict_list})
                save_dict.update({'test_labels': list(y_test)})
                # print accuracies for each stage
                print(' \n Train Acc: ', save_dict['train_PEAKDIST'][-1])
                print(' \n Val Acc: ', save_dict['val_PEAKDIST'][-1])
                print(' \n Test Acc: ', test_acc)

                # SAVING # SAVING # SAVING # SAVING # SAVING # SAVING # SAVING # SAVING 
                save_dict.update({'run time': (time.process_time() - run_start_time)})
                #_save_location = save_dict['save_location']
                #title = save_dict['Run']
                #save2json(save_dict, loop_run_name, _save_location)
                #save2csv(save_dict, title, _save_location)
                model.to('cpu')
                torch.save(model.state_dict(), f"{SS.save_dir}{loop_run_name}.pkl")
                torch.cuda.empty_cache()
    _go()
