import visdom
import time
import numpy as np
#from matplotlib import pyplot as plt
from sklearn.metrics import roc_curve


class Visualizer(object):

    def __init__(self, env='default', **kwargs):
        self.vis = visdom.Visdom(env=env, **kwargs)

        self.iters = {}
        self.lines = {}
        self.plots = {}
        self.env = env

    def display_current_results(self, iters, x, name='train_loss'):
        if name not in self.iters:
            self.iters[name] = []

        if name not in self.lines:
            self.lines[name] = []

        self.iters[name].append(iters)
        self.lines[name].append(x)

        self.vis.line(X=np.array(self.iters[name]),
                      Y=np.array(self.lines[name]),
                      win=name,
                      opts=dict(legend=[name], title=name))

    def display_roc(self, y_true, y_pred):
        fpr, tpr, ths = roc_curve(y_true, y_pred)
        self.vis.line(X=fpr,
                      Y=tpr,
                      # win='roc',
                      opts=dict(legend=['roc'],
                                title='roc'))


    def plot(self, var_name, split_name, title_name, x, y):
            if var_name not in self.plots:

                self.plots[var_name] = self.vis.line(X=np.array([x, x]), Y=np.array([y, y]), win=var_name,env=self.env,
                                                     opts=dict(
                                                         legend=[split_name],
                                                         title=title_name,
                                                         xlabel='Epochs',
                                                         ylabel=var_name
                                                     ))
            else:
                self.vis.line(X=np.array([x]), Y=np.array([y]), env=self.env, win=var_name, name=split_name,
                              update='append')
