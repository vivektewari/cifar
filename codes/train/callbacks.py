import pandas as pd
import torch
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score,mean_squared_error
from codes.utils.visualizer import Visualizer
import os,cv2
from codes.utils.diagnostic import Diagonose
from catalyst.dl  import  Callback, CallbackOrder,Runner
import time
from codes.utils.auxilary import count_parameters

class MetricsCallback(Callback):

    def __init__(self,
                 directory=None,
                 model_name='',
                 check_interval=1,
                 input_key: str = "targets",
                 output_key: str = "logits",
                 prefix: str = "acc_pre_rec_f1",
                 config      =None,

                 ):
        super().__init__(CallbackOrder.Metric)

        self.input_key = input_key
        self.output_key = output_key
        self.prefix = prefix
        self.directory = directory
        self.model_name = model_name
        self.check_interval = check_interval

        self.visualizer = Visualizer()
        self.config=config


    def create_dist_data(self,dict):
        df=pd.DataFrame()
        nas=pd.DataFrame()
        for key in dict.keys():

            na_temp=pd.DataFrame.from_dict({'val':[int(np.isnan(dict[key]).sum()),int((dict[key]==0).sum())],'key':[str(key)+"_na",str(key)+'_zero']})

            temp=pd.DataFrame.from_dict({'val':dict[key],'key':[key for i in range(dict[key].shape[0])]})
            df=df.append(temp.dropna())
            nas=nas.append(na_temp)
        return df,nas.set_index('key')
    def get_diagnostics(self,model,input1,target,loss_func,save_loc,vis,step):
        """
        grad_dict:float in visdom,saves_grad_dict_distribution,saves activation_dict distribution,saves output_distribution

        :return:
        """
        d=Diagonose()
        output=d.get_everything(model,input1,target,loss_func)
        max_vals, max_indices = torch.max(output['model_output'], 1)

        #drawings
        for key in output['perc_98_grad'].keys():
            vis.plot('perc_98_grad', str(key), 'perc_98_grad', step,output['perc_98_grad'][key] )
        #vis.display_current_results(step, output['perc_98_grad'],name='train_grad_98')

        plt_names=['gra_plot','layer_plot','output_vs_target_plot']
        datas = [output['grad_dict'],output['r_dict']]
        for i in range(3):
            if i <2:
                d,nas=self.create_dist_data(datas[i])
                g = sns.FacetGrid(d,  row="key",height=4, aspect=1.5,sharex=True,sharey=False)

                g.map(sns.kdeplot, "val")
                plt.figtext(0.5, 0.9, 'Na,zero count: {}'.format(nas.to_dict()['val']), wrap=True, horizontalalignment='center', fontsize=12)



            else:
                temp=pd.DataFrame.from_dict({'val':max_indices,'key':['pred' for i in range(len(target))]})
                temp2 = pd.DataFrame.from_dict(
                    { 'val': target, 'key': ['actual' for i in range(len(target))]})

                d=temp.append(temp2)
                sns.displot(data=d,hue='key',x='val', kind="kde")
            plt.savefig(save_loc+plt_names[i]+str(step)+'.jpg')
            plt.close()


    def getMetrics(self,actual, predicted):
        """
        :param actual: actual series
        :param predicted: predicted series
        :return: list of accuracy ,precision ,recall and f1
        """
        met = []
        metrics = ['accuracy_score','classification_error']#, precision_score, recall_score, f1_score]
        for m in metrics:
            if m == 'accuracy_score':
                met.append(np.diag(predicted[:,actual]).mean())
            elif m=='classification_error':
                max_vals, max_indices = torch.max(predicted, 1)
                train_error = (max_indices != actual).sum()/ max_indices.size()[0]
                met.append(train_error)
        return met


    def on_loader_end(self, state: "Runner") -> None:
        if state.is_train_loader:
            if (state.epoch_step - 1) % 50 == 0:
                self.get_diagnostics(state.model, state.batch['indep'], state.batch['targets'],
                                     state.criterion, self.directory, self.visualizer, state.epoch_step)

            met = self.getMetrics(state.batch['targets'], state.batch['logits'].detach())
            self.visualizer.display_current_results(state.epoch_step, met[1],
                                            name='train_classification_error')
    def on_epoch_end(self, state: Runner):

        """Event handler for epoch end.

        Args:
            runner ("IRunner"): IRunner instance.
        """
        #print(torch.sum(state.models.fc1.weight),state.models.fc1.weight[5][300])
        #print(torch.sum(state.models.conv_blocks[0].conv1.weight))
        if self.directory is not None and state.epoch_step%100==0:
            torch.save(state.model.state_dict(), str(self.directory) + '/' +
                self.model_name + "_" + str(state.epoch_step) + ".pth")
            # pd.DataFrame(data={'targets':state.batch['targets'],'pred':state.batch['logits']},
            #              index=[i for i in range(len(state.batch['targets']))]).to_csv(self.directory+'pred_vs_actual'+str(state.epoch_step)+'.csv')

        if (state.epoch_step + 1) % self.check_interval == 0:
            met = self.getMetrics(state.batch['targets'], state.batch['logits'])

            self.visualizer.display_current_results(state.epoch_step, state.epoch_metrics['train']['loss'],
                                                    name='train_loss')
            self.visualizer.display_current_results(state.epoch_step, state.epoch_metrics['valid']['loss'],
                                                    name='valid_loss')
            self.visualizer.display_current_results(state.epoch_step, met[0],
                                                    name='valid_accuracy')
            self.visualizer.display_current_results(state.epoch_step, met[1],
                                                    name='valid_classification_error')
        if state.epoch_step==1:
            self.start_time=time.time()
            self.table=count_parameters(state.model)
        if state.epoch_step%100 == 0:
            time_passed=time.time()-self.start_time
            with open(self.directory+'summary.txt', 'w') as w:
                w.write(self.config.model_name)
                w.write('\n')
                w.write(self.table[0].get_string())
                w.write(str(self.table[1]))
                w.write('\n')
                w.write('epoch{},Time_taken:{},val_classification_error:{},learning_rate:{}'
                        .format(str(state.epoch_step),str(time_passed),str(met[1]),self.config.learning_rate))
