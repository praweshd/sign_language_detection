import os
import shutil

user = '/home/ecbm6040/dataset5/'
f= os.listdir(user)
print(f)

# labels = []
# names =[]
# train_path = os.path.join('dataset5/','data_A')
# if not os.path.exists(train_path):
#     os.makedirs(train_path)
# count=0
# for a in f:
#     path = os.path.join(user,a)
#     b =  os.listdir(path)
    
#     for x in b:
#         if(x.split('_')[0]=='color'):
#             labels.append(count)
#             names.append(x)

#             # print(x)
#             old_path = os.path.join(path,x)
#             new_im_path = os.path.join(train_path,x)
#             #print(old_path,new_im_path)
#             shutil.move(old_path,new_im_path)
#     count = count+1
# cs = [names,labels]
# import csv

# with open('dataset5/labels_A.csv','w') as f1:
#     writer = csv.writer(f1)
#     writer.writerows(zip(names, labels))
