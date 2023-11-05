

import numpy as np

class Diagonose(object):
    """

    """
    def select_layers(self,model):
        """
        Selects 3 layers 1,[last_layer/2].last
        :param model:
        :return:
        """
        layers=[0,int(np.floor(model.num_layers/2)),model.num_layers-1]
        if layers[1]==0:layers.remove(layers[1])
        return layers
    @staticmethod
    def representation_info(model,layers,input):
        r_dict= {}

        for i in layers:
            r_dict[i]= model.run_till(input,i).detach().flatten()
        return r_dict
    def weight_info(self,model,layers:list[int]) :#changes with choosen models toDo:change this to gnerelize ones
        w_dict,grad_dict={},{}

        for i in layers:
            w_dict[i]=model.l[i].weight.detach().flatten()
            grad_dict[i]=model.l[i].weight.grad.detach().flatten()

        return w_dict,grad_dict

    @staticmethod
    def zeros_non_zeros(list_:list,threshold:float=0.00000000000001):
        temp=np.concatenate(list_,axis=0)
        temp=abs(temp)
        size=len(temp)
        temp_zero=temp[(temp<=threshold)]
        temp_non_zero=temp[(temp>threshold)]
        return temp_zero,temp_non_zero,size



    @staticmethod
    def get_quantiles_for_non_zeros(input_:np.array):
        quantiles=[]
        for q in [0.20,0.5,0.8]:
            quantiles.append(np.quantile(input_,q))
        return quantiles
    def get_everything(self,model,input,target,loss_func):
        layers=self.select_layers(model) #diag layers
        #run models
        x, y = input, target
        y_hat = model(x)
        loss = loss_func(y_hat, y)
        loss.backward()

        #collect grads and
        w_dict,grad_dict=self.weight_info(model,layers)

        perc_98_grad={}
        for key in grad_dict.keys():
            perc_98_grad[key]=np.quantile(np.abs(grad_dict[key]),0.98)

        r_dict=self.representation_info(model,layers,input)

        return {'weight':w_dict,'grad_dict':grad_dict,'r_dict':r_dict,'perc_98_grad':perc_98_grad,'model_output':y_hat}
if __name__ == '__main__':
    from codes.utils.diagnostic import Diagonose
    from codes import config
    from codes.train.losses import custom_EntropyLoss
    from codes.models.baseline_models import fc_model
    from codes.train.losses import custom_EntropyLoss
    import torch




    d=Diagonose()
    model=fc_model(**config.model_paramsfc1)

    input=torch.ones(3,32,32,3)
    output=d.get_everything(model,input,target=torch.tensor([2,4,6]),loss_func=custom_EntropyLoss())


    k=0