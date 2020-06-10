from extract_ROI import get_masker, get_ROI
import configparser
import subprocess
import pandas as pd
import yaml
import os

def get_hyperparameters(job_id):
    # Update file name with correct path
    with open("./hyperparams.yml", 'r') as stream:
        hyper_param_set = yaml.load(stream)
    
    return hyper_param_set[job_id-1]["hyperparam_set"]

def run_ROI_kubernetes():
    config_path = os.environ['CONFIG_PATH']
    job_id = int(os.environ['JOB_ID'])

    config = configparser.ConfigParser()
    config.read(config_path)
    data_path = config['GENERAL']['data_path']
    atlas_name = config['ATLAS']['atlas_name']
    remove_confounds = bool(config['ATLAS']['remove_confounds'])
    n_jobs = int(config['GENERAL']['n_jobs'])
    
    data = pd.read_csv(data_path)
    people_path = data['path']
    people_number = data['people_number']

    person_path = data.loc[job_id]['path']
    person_number = data.loc[job_id]['people_number']
    masker = get_masker(atlas_name)
    data = get_ROI(person_path, masker, remove_confounds)
    columns = ['x' + str(i) for i in range(data.shape[1])]
    data = pd.DataFrame(data, columns=columns)
    save_path = config['ATLAS']['path_to_write'] + str(person_number)+'.csv'
    if config['ATLAS']['where_to_write']=='hdd':
        data.to_csv(save_path, index=False)
    else:
        #TODO database
        pass

    
def extract_ROI_kubernetes(number_of_person):

    bashCommand = "bash create_jobs.sh {0}".format(number_of_person)
    process = subprocess.Popen(bashCommand.split(), stdout=subprocess.PIPE)
    
    bashCommand = "kubectl create -f hyperparam-jobs-spec"
    process = subprocess.Popen(bashCommand.split(), stdout=subprocess.PIPE)
    # output, error = process.communicate()