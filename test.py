import glob
import os
import zipfile
import numpy as np
import pandas as pd



list_files = glob.glob('/data/test/roi/*.csv')
print(len(list_files))

# people = [person.split('/')[-1].split('_')[0] for person in list_files]
# data = pd.DataFrame()
# data['path'] = list_files
# data['people_number'] = people
# data = data.sort_values(by='people_number')
# data.to_csv('./data/data_test.csv', index=False)
# print(len(people))
# print(list_files[0])

# archive = zipfile.ZipFile(list_files[0])
# for i in archive.namelist():
#     print(str(i))

# for file in archive.namelist():
#     if file.startswith('588565/MNINonLinear/Results/rfMRI_REST1_LR/rfMRI_REST1_LR.nii.gz'):
#         archive.extract(file, '/data/test')

