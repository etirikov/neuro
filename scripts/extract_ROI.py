import nibabel
import os
import sys
import glob
import configparser
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

def get_ROI(path, masker, delete_confounds):
    if delete_confounds:
        confounds = image.high_variance_confounds(path, n_confounds = 5)
        return masker.transform(path, confounds = confounds)
    else:
        return masker.transform(path)
    

def get_masker(atlas_name):
    if atlas_name=='harvard_oxford':
        atlas = datasets.fetch_atlas_harvard_oxford('cort-maxprob-thr25-2mm')
    else:
        raise OSError('there is no atlas {0}'.format(atlas_name))
    atlas_filename = atlas.maps

    masker = NiftiLabelsMasker(labels_img=atlas_filename, 
                               standardize=True,
                               memory='nilearn_cache')
    masker.fit()
    return masker


def parallel_jobs_one_machine(config, person_path, person_number, masker, remove_confounds):
    data = get_ROI(person_path, masker, remove_confounds)
    columns = ['x' + str(i) for i in range(data.shape[1])]
    data = pd.DataFrame(data, columns=columns)
    save_path = config['ATLAS']['path_to_write'] + str(person_number)+'.csv'
    if config['ATLAS']['where_to_write']=='hdd':
        data.to_csv(save_path, index=False)
    else:
        #TODO database
        pass


def extract_ROI_one_machine(config_path):
    config = configparser.ConfigParser()
    config.read(config_path)
    data_path = config['GENERAL']['data_path']
    atlas_name = config['ATLAS']['atlas_name']
    remove_confounds = bool(config['ATLAS']['remove_confounds'])
    n_jobs = int(config['GENERAL']['n_jobs'])
    
    data = pd.read_csv(data_path)
    people_path = data['path']
    people_number = data['people_number']
    
    #atlas
    masker = get_masker(atlas_name)

    if n_jobs <= 1:    
        for person in zip(people_path, people_number):
            person_path = person[0]
            person_number = person[1]
            data = get_ROI(person_path, masker, remove_confounds)
            columns = ['x' + str(i) for i in range(data.shape[1])]
            data = pd.DataFrame(data, columns=columns)
            save_path = config['ATLAS']['path_to_write'] + str(person_number)+'.csv'
            if config['ATLAS']['where_to_write']=='hdd':
                data.to_csv(save_path, index=False)
            else:
                #TODO database
                pass
    else:
        params = list(zip([config]*data.shape[0],
                          people_path, 
                          people_number, 
                          [masker]*data.shape[0], 
                          [remove_confounds]*data.shape[0]))

        with Pool(n_jobs) as pool:
            pool.starmap(parallel_jobs_one_machine, params)

def extract_ROI_spark(config_path):
    pass
