import pandas as pd
import random
from data_loaders import *
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm
from torch.utils.data import DataLoader
import torch
from codes.models.baseline_models import FeatureExtractor_baseline, fc_model, ResNet
from data_loaders import cifarDataset
random.seed(24)

def get_top(pred,actual,ind=1):
    top_i_idx = np.argsort(pred,axis=1)[:,-ind:]
    #top_i_values = [pred[i] for i in top_i_idx]
    output=[]
    for i in range(top_i_idx.shape[0]):
        if np.any(actual[i]==top_i_idx[i]):output.append(1)
        else :output.append(0)
    return output

def get_inference(model,dataset,notes= ""):
    """
    algo:1 model output. Output, top 1 and 3  classification tagging
     pred:Takes dataloader,model,saved weight loc as input, produce pred. creates df and save in excel.

    """



    output=pd.DataFrame()
    obs_count=len(dataset.images)
    k=0
    while k < obs_count:
        j=min(k+256,obs_count)
        pred = model(torch.stack(dataset.images[k:j])).detach().numpy()
        actuals = [int(dataset.labels[k+i]) for i in range(len(dataset.labels[k:j]))] # removing tensor
        top_1=get_top(pred,actuals,ind=1)
        top_3 = get_top(pred, actuals, ind=3)
        prob_act=[pred[i,actuals[i]] for i in range(len(actuals))]
        pred_ind,prob=np.argmax(pred, axis=1),np.max(pred, axis=1)

        df = pd.DataFrame({'image_index':dataset.image_identifier[k:j],'actual': actuals,
                                 'pred':pred_ind,'prob':prob,'prob_act':prob_act, 'top_1':top_1,'top_3':top_3 })
        output=output.append(df)
        k = j
        return output


if __name__ == "__main__":
    from codes.models.baseline_models import FeatureExtractor_baseline,fc_model,ResNet
    from data_loaders import cifarDataset
    from types import SimpleNamespace
    import yaml
    from codes.utils.diagnostic import Diagonose
    with open('../train/config.yaml', 'r') as f:
            config = SimpleNamespace(**yaml.safe_load(f))
    from codes.train.losses import  custom_EntropyLoss

    model=ResNet(**config.model_paramsrn1)
    checkpoint = torch.load(config.pretrained,map_location=torch.device('cpu'))
    model.load_state_dict(checkpoint, strict=True)
    model.eval()
    data_loader=cifarDataset
    config.data_loader_param_v['max_rows']=10000
    dataset= data_loader(**config.data_loader_param_v)
    loc='/home/pooja/PycharmProjects/cifar/weight/'
    df=get_inference(model,dataset)
    df.sort_values('prob_act').to_csv(loc+'model_output.csv')
    # Diagonose.get_weight_image(model,loc)
    # #output_images of succesful prediction
    # best=df.sort_values(['prob_act'],ascending=False).drop_duplicates(['actual']).reset_index()
    # for i in range(best.shape[0]):
    #     index=dataset.image_identifier.index(best.iloc[i]['image_index'])
    #     image_name=str(best.iloc[i]['image_index'])+"_"+str(int(dataset.labels[index]))
    #     Diagonose.representation_info_plots(model,torch.stack([dataset.images[index]]),notes=image_name,loc=loc)
