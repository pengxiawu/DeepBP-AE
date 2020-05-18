import os
import numpy as np

res = "/home/lab2255/Myresult/csic_res/20200516_deepMIMOdataset_l1gaec"

os.listdir(res)

files = [file for file in os.listdir(res) if file[-3:] == 'npy' and file[:3]=='res']
files = sorted(files, key=lambda x: x.split('.', 1)[0])
print(files)


for i in range(len(files)):
    print ('processing: ', files[i])
    temp_file = np.load(os.path.join(res, files[i]), encoding='latin1',allow_pickle=True)
    if i == 0:
        result_dict = temp_file.item()
        print(result_dict)
    else:
        for key in result_dict.keys():
            result_dict[key].append(temp_file.item()[key][0])

print(result_dict)


