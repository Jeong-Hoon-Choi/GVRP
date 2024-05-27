import pandas as pd
from tqdm import tqdm
import io
import numpy as np
from tabulate import tabulate
from ffn import *
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
import pickle
from ftt import *
import tensorflow as tf
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc
import time

gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
  try:
    tf.config.experimental.set_visible_devices(gpus[0], 'GPU')
  except RuntimeError as e:
    print(e)

vcf_dir = '../../../data/Human/vcf/'
data_dir = '../../../data/Human/csv/'
model_dir = '../../../model/'
performance_dir = '../../../result/Human/GVRP_refinement_model/csv/'
roc_dir = '../../../result/Human/ROC/'

data_type_list = ['ht.indel.csv', 'ht.snp.csv', 'hm.indel.csv', 'hm.snp.csv']
mode = ['train_HG001_test_HG002', 'train_HG002_test_HG001', 'mixed_data']
# mode = ['concat_data_train_HG001_test_HG002', 'concat_data_train_HG002_test_HG001', 'mix_all']

models = ['XG_Boost', 'LGBM', 'RF', 'LR', 'KNN', 'NB', 'MLP', 'ft_transformer']

drop_features = ['label']
label = 'label'
