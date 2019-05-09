import os
import shutil
import csv


root = '/home/ecbm6040/dataset5/'
f = os.listdir(root)
print(f)


for person in f:

	labels = []
	names =[]

	dest_path = os.path.join('/home/ecbm6040', 'dataset', person)
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
	            labels.append(count)
	            names.append(x)

	            # print(x)
	            old_path = os.path.join(path, x)
	            new_im_path = os.path.join(dest_path, x)
	            #print(old_path,new_im_path)
	            shutil.move(old_path, new_im_path)
	    count = count+1
	cs = [names,labels]

	file_name = '/home/ecbm6040/dataset/' + person + '.csv'
	with open(file_name,'w') as f1:
	    writer = csv.writer(f1)
	    writer.writerows(zip(names, labels))
