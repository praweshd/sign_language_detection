import os
import shutil
import csv


root = '/home/ecbm6040/dataset5/'
f = os.listdir(root)
print(f)


labels = []
names =[]

for person in f:

	dest_path = os.path.join('/home/ecbm6040', 'dataset')
	if not os.path.exists(dest_path):
	    os.makedirs(dest_path)
	
	src_path = os.path.join(root, person)
	g = os.listdir(src_path)
	
	count=0
	
	for letters in g:
	    path = os.path.join(src_path, letters)
	    b =  os.listdir(path)
	    
	    for x in b:
	        if(x.split('_')[0]=='color'):
	            labels.append(x.split('_')[1])
	            names.append(person + '_' + x)

	            # print(x)
	            old_path = os.path.join(path, x)
	            new_im_path = os.path.join(dest_path, x)
	            #print(old_path,new_im_path)
	            shutil.move(old_path, new_im_path)
	            os.rename(new_im_path, os.path.join(dest_path, person + '_' + x ))
	    count = count+1

file_name = '/home/ecbm6040/dataset/labels.csv'
with open(file_name,'w') as f1:
    writer = csv.writer(f1)
    writer.writerows(zip(names, labels))
