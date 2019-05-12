import os
import shutil
import csv
import pandas as pd 

train = pd.csv_read('/home/ecbm6040/dataset_update/train_1.csv')
val = pd.csv_read('/home/ecbm6040/dataset_update/val_1.csv')
test = pd.csv_read('/home/ecbm6040/dataset_update/test_1.csv')

train = train.sample(frac=1)
print(train)
