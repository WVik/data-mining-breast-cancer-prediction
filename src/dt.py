import numpy as np
from sklearn import preprocessing, model_selection, neighbors
import pandas as pd


dataFrame = pd.read_csv('/Users/Vikram/Documents/Projects/Data-Mining-Breast-Cancer/data/breast-cancer-wisconsin.data',
                       names = ['id','clump_thickness', "unif_cell_size", "unif_cell_shape", "marg_adhesion", "single_epith_cell_size", "bare_nuclei", "bland_chrom", "norm_nucleoli", "mitoses", "class"])

dataFrame.replace('?','-9999999', inplace=True)
dataFrame.drop(['id'],1,inplace=True)