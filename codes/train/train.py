import random

from data_loaders import *
from catalyst.dl import SupervisedRunner

from codes.models.baseline_models import count_parameters

import torch.optim as optim
from torch.utils.data import DataLoader
import pandas as pd
from catalyst import dl
#from callbacks import MetricsCallback
from sklearn.model_selection import StratifiedKFold
import torch
def train(model_param,model_,data_loader_param,data_loader_param_v=None,data_loader=None,loss_func=None,callbacks=None,param=None):

    data_load = data_loader(**data_loader_param)
    criterion = loss_func
    model = model_(**model_param)
    count_parameters(model)
    # models = FCLayered(**get_dict_from_class(model_param,models))
    if param.pretrained is not None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        checkpoint = torch.load(param.pretrained, map_location=device)
        try:
            model.load_state_dict(checkpoint['model_state_dict'])
        except:
            model.load_state_dict(checkpoint)
        model.eval()
    optimizer = optim.SGD(model.parameters(), lr=param.learning_rate,momentum=param.momentum,weight_decay=param.weight_decay)

    # models.clear_rc_seq()

    if data_loader_param_v is None:
        train_len = data_load.__len__()

        train_set=set(np.random.choice([i for i in range(train_len)],size=int(0.8*train_len),replace=False))
        valid_set=set([i for i in range(train_len)]).difference(train_set)
        print("train samples:{}   valid samples:{}".format(len(train_set),len(valid_set)))

        data_loader_param['indexes']=list(train_set)
        data_loader_param_v = data_loader_param
        data_loader_param_v['indexes'] =  list(valid_set)


    loaders = {
        "train": DataLoader(data_loader(**data_loader_param),
                            batch_size=2048,
                            shuffle=False,
                            num_workers=4,
                            pin_memory=True,
                            drop_last=False),
        "valid": DataLoader(data_loader(**data_loader_param_v),
                            batch_size=4096,
                            shuffle=False,
                            num_workers=4,
                            pin_memory=True,
                            drop_last=False)
    }

    callbacks = callbacks
    runner = SupervisedRunner(

        output_key="logits",
        input_key="indep",
        target_key="targets")
    # scheduler=scheduler,

    runner.train(
        model=model,
        criterion=criterion,
        loaders=loaders,
        optimizer=optimizer,

        num_epochs=1000,
        verbose=True,
        logdir=f"fold0",
        callbacks=callbacks,
    )

    # main_metric = "epoch_f1",
    # minimize_metric = False

if __name__ == "__main__":
    from callbacks import MetricsCallback
    from codes.models.baseline_models import FeatureExtractor_baseline,fc_model,ResNet
    from data_loaders import cifarDataset
    from  torch.nn import CrossEntropyLoss

    from types import SimpleNamespace
    import yaml



    with open('../train/config.yaml', 'r') as f:
            config = SimpleNamespace(**yaml.safe_load(f))
    from codes.train.losses import  custom_EntropyLoss




    callbacks = [MetricsCallback(input_key="targets", output_key="logits",
                         directory=config.rootdir+config.weightdir, model_name=config.model_name,config=config)]



    train(model_param=config.model_params2,model_=FeatureExtractor_baseline,data_loader_param=config.data_loader_param
          ,data_loader_param_v=config.data_loader_param_v,data_loader=cifarDataset,
          loss_func=custom_EntropyLoss(),callbacks=callbacks,param=config)