import os
import shutil
import csv
import pandas as pd 

train = pd.read_csv('/home/ecbm6040/dataset_update/train.csv')
val = pd.read_csv('/home/ecbm6040/dataset_update/val.csv')
test = pd.read_csv('/home/ecbm6040/dataset_update/test.csv')

# train = train.sample(frac=1)
# val = val.sample(frac=1)
# test = test.sample(frac=1)

# train.to_csv('/home/ecbm6040/dataset_update/train.csv', index = False)
# val.to_csv('/home/ecbm6040/dataset_update/val.csv', index = False)
# test.to_csv('/home/ecbm6040/dataset_update/test.csv', index = False)
# train.to_csv('/home/ecbm6040/dataset_update/train.csv')
# train.to_csv('/home/ecbm6040/dataset_update/train.csv')

# print(train.loc[train.Label == 24])
# train.loc[train.Label == 24].Label = 9
train = train.replace(24, 9)
val = val.replace(24, 9)
test = test.replace(24, 9)
train.to_csv('/home/ecbm6040/dataset_update/train_2.csv', index = False)
val.to_csv('/home/ecbm6040/dataset_update/val_2.csv', index = False)
test.to_csv('/home/ecbm6040/dataset_update/test_2.csv', index = False)