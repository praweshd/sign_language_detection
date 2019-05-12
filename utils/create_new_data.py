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

print(train[label] == 24)