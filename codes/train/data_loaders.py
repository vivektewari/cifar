from torch.utils.data import Dataset
import pandas as pd
import os
import torch



import numpy as np
import os,cv2



class cifarDataset(Dataset): # borrowed from riid challange work
    def __init__(self,loc,indexes=None,max_rows =100000000):  # HDKIM 100
        super(cifarDataset).__init__()
        #loc='/home/pooja/PycharmProjects/lux_ai/outputs/inputs/'


        self.max_rows=max_rows
        self.images,self.labels=self.get_images(loc)
        if indexes is not None:
            self.images,self.labels=[self.images[i] for i in indexes],[self.labels[i] for i in indexes]
        print("data len:{}   data mean:{}  data std:{}".format(len(self.images),torch.mean(torch.concat(self.images).flatten()),torch.std(torch.concat(self.images).flatten())))



    def get_images(self,loc):
        """
        walk through folder   collects label and data  swap data axes  converting to tensor
        appending to list
        """
        img,lab=[],[]
        count=0
        for _, _, files in os.walk(loc):
            for file in files:
                label=file.split("_")[-1]
                image_data=np.swapaxes(cv2.imread(loc+file),0,2)
                img.append(torch.tensor(image_data)/256.0)
                lab.append(torch.tensor(int(label[0])))
                count+=1
                if count>self.max_rows:break
        return img,lab



    def __len__(self):
        #return len(self.dict.keys()) #earlier
        return len(self.labels)


    def __getitem__(self, index):
        return {'indep':self.images[index],'targets':self.labels[index]}

if __name__ == "__main__":pass
    # from funcs import get_dict_from_class
