import os
import sys
import glob
import configparser
import subprocess
import socket
from multiprocessing.pool import Pool
import shutil

import numpy as np
import pandas as pd

from nilearn.input_data import NiftiLabelsMasker
from nilearn import image
from nilearn.connectome import ConnectivityMeasure
from nilearn import datasets
import nibabel

from scipy.signal import savgol_filter, medfilt

from pyspark.sql.functions import col, pandas_udf, PandasUDFType
from pyspark.sql.types import NullType
from pyspark.sql import SparkSession
from pyspark import SparkConf
conf = SparkConf()
# conf.setMaster('yarn-client')
conf.setAppName('Test')
conf.set("spark.hadoop.yarn.resourcemanager.hostname", '172.17.14.17')
conf.set("spark.hadoop.yarn.resourcemanager.address", '172.17.14.17:8050')
conf.set('spark.executor.memoryOverhead', '10GB')
conf.set('spark.executor.instances', '12')
spark = SparkSession.builder.config(conf=conf).getOrCreate()

def get_ROI(path, masker, delete_confounds):
    if delete_confounds:
        confounds = image.high_variance_confounds(path, n_confounds = 5)
        return masker.transform(path, confounds = confounds)
    else:
        return masker.transform(path)
    

def get_masker(atlas_name, atlas_path):
    if atlas_name=='harvard_oxford':
        atlas = datasets.fetch_atlas_harvard_oxford('cort-maxprob-thr25-2mm', data_dir=atlas_path)
    else:
        raise OSError('there is no atlas {0}'.format(atlas_name))
    atlas_filename = atlas.maps

    masker = NiftiLabelsMasker(labels_img=atlas_filename, 
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
    print('start extract')
    if n_jobs <= 1:    
        for person in zip(people_path, people_number):
            print(person)
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


@pandas_udf("path string, people_number long", PandasUDFType.GROUPED_MAP)
def extract_ROI_spark(data):
    print('111', data.shape)
    people_path = data['path']
    people_number = np.unique(data['people_number'])[0]

    atlas_path = '/data/test/nilearn_data'
    
    masker = get_masker(atlas_name, atlas_path)
    j = 0
    for person in zip(people_path, [people_number]*data.shape[1]):
        person_path = person[0]
        person_number = person[1]
        print(person_path, person_number)
        data_person = get_ROI(person_path, masker, remove_confounds)
        columns = ['x' + str(i) for i in range(data_person.shape[1])]
        data_person = pd.DataFrame(data_person, columns=columns)
        save_path = config['ATLAS']['path_to_write'] + str(person_number)+'_{}_.csv'.format(j)
        j+=1
        if config['ATLAS']['where_to_write']=='hdd':
            data_person.to_csv(save_path, index=False)

    temp = pd.DataFrame([[j,people_number]], columns=['path', 'people_number'])
    return temp

if __name__ == '__main__':
    config_path = './config.ini'
    config = configparser.ConfigParser()
    config.read(config_path)

    data_path = config['DATA']['data_path']
    atlas_name = config['ATLAS']['name']
    remove_confounds = bool(config['ATLAS']['remove_confounds'])
    n_jobs = int(config['GENERAL']['n_jobs'])

    masker = get_masker(atlas_name, None)
    
    data = pd.read_csv(data_path)
    data_spark = spark.createDataFrame(data)

    df1 = data_spark.groupby("people_number").apply(extract_ROI_spark)
    print(df1.select('path').toPandas())
