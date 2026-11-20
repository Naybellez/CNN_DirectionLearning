# last edited 191126 - tidying

# Script containing functions handling the training, vlaidation and test loops
# Script also contains model performance metrics functions

# IMPORTS
import numpy as np
from tqdm import tqdm
import random

import torch
from torch.utils.data import DataLoader

import sys
sys.path.append('../.')
from src.dataloaderP3Direction import IDSWDataSetLoader3
from src.functions import ImageProcessor


# ASSESSMENT functions
# MSE # same as loss but this is to be held on to for human eyes
def MSE_metric(preds, labels):
    # Returns Mean Squared Error
    return torch.mean((preds-labels)**2).item()

# MAE # similar to above but absolute error. may provide wider understanding
def MAE_metric(preds, labels):
    # Returns Mean Absolute Error
    return torch.mean(torch.abs(preds-labels)).item()

def MAE_metric2(preds, labels, num_classes = 360):
    # Another calculation for Mean Absolute Error- no differnce to above
    return torch.minimum(torch.mean(torch.abs(preds-labels)).item(), num_classes - torch.mean(torch.abs(preds-labels)).item())
    
# peak distance error. # distance between the two gaus peaks (one for true labels and one for predictions)
def peak_disterr_metric2(preds, labels):
    """
    # Peak Distance Error. Custom function for guassian distributed labels
    # distance between the two gaus peaks (one for true labels and one for predictions)
    Return mean peak distance for input batch of predictions and labels
    """
    pred_idx = torch.argmax(preds, dim=1).float()
    labels_idx = torch.argmax(labels, dim=1).float()
    num_classes = labels.shape[1]
    
    # Absolute difference
    diff = torch.abs(pred_idx - labels_idx)
    # Wrap around circle
    wrapped_diff = torch.minimum(diff, num_classes - diff) # to account for circular labels. return the smaller val (diff OR num_classes- diff)
    
    return [w.item() for w in wrapped_diff], torch.mean(wrapped_diff).item()


def get_roughAcc(Tlabel, Preds):
    """
    A rough calculation of accuracy
    Acuracy is counted if the peak (argmax) of the prediction lies within the True Label range
    NOT increased with proximity to True Label peak (argmax)
    """
    num_tries = 0
    num_correct = 0
    Tlabel = Tlabel.to('cpu')
    Preds = Preds.to('cpu')
    for j in range(len(Tlabel)): 
        labelrange = np.argwhere(Tlabel[j] > 0)
        if Preds[j].argmax() in labelrange:
            num_correct += 1
        num_tries += 1
    acc = (num_tries, num_correct, (num_correct/num_tries)*100)
    return acc

# LOOP FUNCTIONS
def loop_batch(model, 
               data, 
               loss_fn,
               device,
               SS,
               optimizer=None, 
               train=True):	# Train and Val loops. Default is train
    """
    Handles individual epoch loop
    Handles one loop for training or validation
    Therefore called twice per epoch in train_val_batch
    """
    model = model
    if train:
        model.train()
    else:
        model.eval()   

    predict_list = []
    total_count = 0
    current_loss = 0
    labels = []
    batch_acc_MSE = []
    batch_acc_MAE = []
    batch_peakdist = []
    img_batch = None
    imNorm_batch = None
    numBatch = 0
    sizeBatch = 0
    
    for i, batch in enumerate(data,0): # loop through batch
        x_batch, y_batch, img_batch, imNorm_batch = batch # unpack batch

        numBatch = len(data) # length of batch
        sizeBatch += len(x_batch)
        if sizeBatch == 0 or numBatch == 0:
            print(f"{i} sizeBatch: {sizeBatch}   numBatch:  {numBatch}")

        prediction = model.forward(x_batch.to(device))
        loss = loss_fn(prediction, y_batch.to(device))
        if train:
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
        acc = get_roughAcc(y_batch, prediction) # accuracy calculation

        # detach data for saving
        [predict_list.append(pred.detach().to('cpu').numpy()) for pred in prediction]
        [labels.append(y.detach().to('cpu').numpy()) for y in y_batch]

        total_count += SS.batchsize
        current_loss += loss.item()

        # Calc metrics and append to lists
        acc_MSE = MSE_metric(prediction.to('cpu'), y_batch.to('cpu'))
        batch_acc_MSE.append(acc_MSE)
        acc_MAE =  MAE_metric(prediction.to('cpu'), y_batch.to('cpu'))
        batch_acc_MAE.append(acc_MAE)
        peakdist, peakdistMEAN = peak_disterr_metric2(prediction.to('cpu'), y_batch.to('cpu'))
        batch_peakdist.append(peakdistMEAN)
        

    if sizeBatch == 0 or numBatch == 0:
        print(f" sizeBatch: {sizeBatch}   numBatch:  {numBatch}")
    sizeBatch = sizeBatch / numBatch # get the average batch size

    if len(batch_acc_MSE) != numBatch:
        print(f"You're maths logic was faulty!", numBatch, len(batch_acc_MSE))
        print(batch_acc_MSE)
    # Calculate means
    batch_acc_MSE_mean = (sum(batch_acc_MSE) / len(batch_acc_MSE))
    batch_acc_MAE_mean = (sum(batch_acc_MAE) / len(batch_acc_MAE))
    batch_peakdist_mean = (sum(batch_peakdist) / len(batch_peakdist))
    print(f"peak dist means  {batch_peakdist_mean}")
    # contain accuracy and metric scores in single dictionary to be returned
    accs = {'baseAcc': acc[2],'MSE':batch_acc_MSE_mean, 'MAE':batch_acc_MAE_mean, 'peakDist':batch_peakdist_mean}
    
    if train:
        return current_loss, predict_list, labels, accs, model, optimizer, img_batch, imNorm_batch
    else:
        return current_loss, predict_list, labels, accs, img_batch, imNorm_batch # changed y_batch to labels in return 


def test_loop_batch(model,data, device, SS, runname="", save_loc =""):
    # Handles one testing loop for batched data
    import sys
    from src.plottingP3Direction import plot_predictions
    
    model = model.eval()
    predict_list = []
    label_list = []
    total_count =0
    baseacc_list = []
    MSE_list = []
    MAE_list = []
    peakdist_list = []

    print("test len data", len(data))
    with torch.no_grad():
        for i, batch in enumerate(data,0): # loop though test data
            tense, label, img_batch, imNorm_batch = batch 
            prediction = model.forward(tense.to(device))

            tacc = get_roughAcc(label, prediction)
            [predict_list.append(pred.to('cpu')) for pred in prediction] 
            [label_list.append(lab.to('cpu')) for lab in label]
            
            total_count += SS.batchsize

            # save images
            IP = ImageProcessor(device)
            IP.view2(img_batch[0], 1, f"{i}_imgOG", save_loc)
            IP.view2(imNorm_batch[0], 1, f"{i}_imgNorm", save_loc)
        
            test_err_MSE = MSE_metric(prediction.to('cpu'), label.to('cpu'))
            test_err_MAE =  MAE_metric(prediction.to('cpu'), label.to('cpu'))
            test_peakdist, testpeakdistMEAN = peak_disterr_metric2(prediction.to('cpu'), label.to('cpu'))

            MSE_list.append(test_err_MSE)
            MAE_list.append(test_err_MAE)
            peakdist_list.append(testpeakdistMEAN)
            baseacc_list.append(tacc[2])

        # plot label and prediction
        plot_predictions(prediction, label, test_peakdist, runname=runname, save_loc=save_loc)

        accuracy = {'BaseAcc':baseacc_list,'MSE': MSE_list, 'MAE': MAE_list, 'peakDist': peakdist_list}
        return accuracy, predict_list, label_list

def train_val_batch(model, train, val, save_dict, loss_fn, epochs, optimizer, device, config, SS): #train_dl, val_dl, 
    """
    Handles the training and validation loops
    Calls loop_batch function appropriately

    """
    import sys
    sys.path.append('../.')
    IP = ImageProcessor(device)

    # to reduce number of var changes later on
    x_train, resolution, av_lum, model_name, half_ciprange, std_dev, batchsize = train
    x_val  = val
    
    t_loss_list = []
    v_loss_list = []
    
    t_predict_list = []
    v_predict_list = []
    
    t_accuracy_MSE_list = []
    t_accuracy_MAE_list = []
    t_accuracy_BASE_list = []
    t_accuracy_PEAKDIST_list = []
    
    v_accuracy_MSE_list = []
    v_accuracy_MAE_list = []
    v_accuracy_BASE_list = []
    v_accuracy_PEAKDIST_list = []
    
    t_label_list = []
    v_label_list = []
        
    total_epochs = 0
        
    for epoch in tqdm(range(save_dict['start_epoch'], epochs)):
        print(f"Data Loading...")
        train_ds = IDSWDataSetLoader3(x_train, resolution, av_lum, model_name, half_ciprange, std_dev, device)
        trainL = DataLoader(train_ds, batch_size=SS.batchsize, shuffle=True, drop_last=True) 

        val_ds = IDSWDataSetLoader3(x_val, resolution, av_lum, model_name, half_ciprange, std_dev, device)
        val = DataLoader(val_ds, batch_size=SS.batchsize, shuffle=True, drop_last=True)

        random_value = random.randrange(0,SS.batchsize)
        print('Training...')

        t_loss, train_prediction, t_label_list, tacc, model, optimizer, img_batch, imNorm_batch = loop_batch(model, 
                                                                                                             trainL,
                                                                                                             loss_fn,
                                                                                                             device,
                                                                                                             config,
                                                                                                             SS,
                                                                                                             optimizer, 
                                                                                                             train=True) 

        print("tacc: ",tacc)
        t_accuracy_MSE_list.append(tacc['baseAcc'])
        t_accuracy_MAE_list.append(tacc['MSE'])
        t_accuracy_BASE_list.append(tacc['MAE'])
        t_accuracy_PEAKDIST_list.append(tacc['peakDist'])
        
        if int(epoch) == int(random_value):
            """
            On a random training epoch
            Display one image per batch - original image and preprocessed image
            Print Prediction and Label for image
            """
            print(f"EPOCH    {epoch} / {epochs}:")
            IP.view2(img_batch[0], 1, "original",save_dict)
            IP.view2(imNorm_batch[0], 1, "Processed",save_dict) # img, scale:int, name:str
            print(f"TRUE LABEL    :          {t_label_list[0]}")
            print(f"PREDICTION    :          {train_prediction[0]}")
        
        t_loss_list.append(t_loss)
        t_predict_list.append(train_prediction)       

        print('Validating...')
        v_loss, val_prediction, v_label_list, vacc, img_batch, imNorm_batch = loop_batch(model, 
                                                                                            val, 
                                                                                            loss_fn,
                                                                                            device,
                                                                                            config,
                                                                                            SS,
                                                                                            optimizer=None, 
                                                                                            train=False)
        print(f"v loss  {v_loss}")
        
        v_accuracy_MSE_list.append(vacc['baseAcc'])
        v_accuracy_MAE_list.append(vacc['MSE'])
        v_accuracy_BASE_list.append(vacc['MAE'])
        v_accuracy_PEAKDIST_list.append(vacc['peakDist'])
            
        v_loss_list.append(v_loss)
        v_predict_list.append(val_prediction)

        total_epochs += 1
        
    save_dict['Current_Epoch'] = epochs
    save_dict['training_samples'] = len(train)
    save_dict['validation_samples'] = len(val)

    save_dict['train_baseAcc'] = t_accuracy_MSE_list
    save_dict['train_MSE'] = t_accuracy_MAE_list
    save_dict['train_MAE'] = t_accuracy_BASE_list
    save_dict['train_PEAKDIST'] = t_accuracy_PEAKDIST_list

    save_dict['val_baseAcc'] = v_accuracy_MSE_list
    save_dict['val_MSE'] = v_accuracy_MAE_list
    save_dict['val_MAE'] = v_accuracy_BASE_list
    save_dict['val_PEAKDIST'] = v_accuracy_PEAKDIST_list

    save_dict['t_loss_list'] = t_loss_list
    save_dict['v_loss_list'] = v_loss_list
    
    save_dict['t_labels'] = t_label_list
    save_dict['v_labels'] = v_label_list
    
    save_dict['t_predict_list'] = t_predict_list 
    save_dict['v_predict_list'] = v_predict_list 
    
    return model, save_dict

