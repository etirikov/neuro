import nibabel
import os
import sys
import glob
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from nilearn.input_data import NiftiLabelsMasker
from nilearn import image
from nilearn.connectome import ConnectivityMeasure
from nilearn import datasets
import seaborn as sns
from multiprocessing.pool import Pool
from scipy.signal import savgol_filter, medfilt
import subprocess

def unzip(people, experiment, data_path):
    bashCommand = "unzip {0}/{1}_3T_rfMRI_REST{2}_fixextended.zip -d ./temp".format(data_path,
                                                                                    people, 
                                                                                    experiment)
    process = subprocess.Popen(bashCommand.split(), stdout=subprocess.PIPE)
    output, error = process.communicate()
    
def delete_folder(people):
    bashCommand = "rm -r ./temp/{}".format(people)
    process = subprocess.Popen(bashCommand.split(), stdout=subprocess.PIPE)
    output, error = process.communicate()
    
def get_ROI(path, masker, delete_confounds):
    if delete_confounds:
        confounds = image.high_variance_confounds(path, n_confounds = 5)
        return masker.transform(path, confounds = confounds)
    else:
        return masker.transform(path)
    

def extract_ROI(data_path, remove_confounds):
    people = []
    for s in sorted(glob.glob('{}/*.zip'.format(data_path)):
        people.append(s.split('/')[-1].split('_')[0])

    people = sorted(np.unique(people).astype(int))

    #atlas
    dataset = datasets.fetch_atlas_harvard_oxford('cort-maxprob-thr25-2mm')
    atlas_filename = dataset.maps

    masker = NiftiLabelsMasker(labels_img=atlas_filename, standardize=True,
                               memory='nilearn_cache')
    path = './temp/{0}/MNINonLinear/Results/rfMRI_REST{1}_{2}/rfMRI_REST{1}_{2}_hp2000_clean.nii.gz'

    masker.fit()
        
    for person in people:
        print(person)
        for experiment in [1, 2]:
            unzip(person, experiment)
            for direction in ['LR', 'RL']:
                try:
                    path_person = path.format(person, experiment, direction)
                    data = get_ROI(path_person, masker, remove_confounds)
                    columns = ['x' + str(i) for i in range(data.shape[1])]
                    data = pd.DataFrame(data, columns=columns)
                    data.to_csv('./all_data_confounds_remove/{0}_REST{1}_{2}.csv'.format(person, 
                                                                                       experiment, 
                                                                                       direction), 
                                index=False)
                except Exception as e:
                    print('error', person)
                    continue
            delete_folder(person)