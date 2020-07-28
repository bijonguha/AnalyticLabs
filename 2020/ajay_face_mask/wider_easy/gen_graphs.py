# -*- coding: utf-8 -*-
"""
Created on Tue Jul 21 00:43:00 2020

@author: BIG1KOR
"""
#%%
import hdf5storage
import pandas as pd

#%%

class CreateData:
    
    def __init__(self, path, name):
        
        self.dict_mat = hdf5storage.loadmat(path)
        self.list_mat = self.dict_mat['pr_cruve']
        
        self.df_tmp = pd.DataFrame(self.list_mat)
        self.df_tmp.columns = ['Precision','Recall']
        self.df_tmp['Model'] = name
        
    def getdf(self):
        return self.df_tmp

#%%
        
mask = CreateData('wider_pr_info_MaskFace_easy_val', 'Mask Face')
df_mask = mask.getdf()

pyramid = CreateData('wider_pr_info_PyramidKey_easy', 'Pyramid Key')
df_pyra = pyramid.getdf()

retina = CreateData('wider_pr_info_RetinaFace_easy', 'Retina Face')
df_retina = retina.getdf()

rcnn = CreateData('wider_pr_info_Face R-CNN_easy', 'Fast Rcnn')
df_rcnn = rcnn.getdf()

alnno = CreateData('wider_pr_info_AInnoFace_easy', 'AlnnoFace')
df_alnno = alnno.getdf()

df_all = pd.concat([df_mask, df_pyra, df_retina, df_rcnn, df_alnno])

#%%

import seaborn as sns; sns.set()
import matplotlib.pyplot as plt

plt.title('Precision - Recall Curves for various architectures')
ax = sns.lineplot(x="Precision", y="Recall", hue="Model",markers=True,
                  data=df_all)