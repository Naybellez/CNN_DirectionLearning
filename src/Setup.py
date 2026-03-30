# Converting pynib run to py file run

# IMPORTS
import wandb
import sys
sys.path.append('../../.')
# CUSTOM FUNCTION IMPORTS

#import SimulationSettings.Settings0 as SS
from src.DL_GO import run_go


def setup(GPU):
    import torch
    wandb.init(mode='offline')
    
    if GPU == 0:
        device = "cuda:0" if torch.cuda.is_available() else "cpu"
        import SimulationSettings.Settings0 as SS 

    elif GPU == 1:
        device = "cuda:1" if torch.cuda.is_available() else "cpu"
        import SimulationSettings.Settings1 as SS

    print("Setup: ",GPU,  device)
    print(SS.model_name)
    
    config = dict({'name': f"Sweep on {SS.model_name} at {SS.resolutioncard[0]['resolution']}_clip:{SS.half_ciprange}"}) #config = dict({'name': 'Sweep on 2C'})

    config.update({"method": "bayes", "metric":{"goal": "minimize", "name": "t_loss"},
                "parameters": {"epochs" :{"value" : SS.epochs},
                                "batch_size": {"value": SS.batchsize},
                                "learning_rate":{"value": SS.learning_rate},
                                "loss_fn_cards": {"value":SS.loss_fn},
                                "optimiser": {"value": SS.optim},
                                "seeds": {"values": SS.seeds},
                                "half_ciprange": {"value":SS.half_ciprange},
                                "std_dev":{"value":SS.std_dev},
                                'model_cards':{"value":SS.modelcard},
                                'resolution_cards':{"value":SS.resolutioncard}
                                }})

    sweep_id = wandb.sweep(sweep= config, project=SS.projectNAME)

    run_go(GPU)
#if __name__ == '__main__':
#	#wandb.agent(sweep_id, function=_go, count=20)

#count = 0
#while count <=len(SS.seeds):
#    _go()
#    count += 1
