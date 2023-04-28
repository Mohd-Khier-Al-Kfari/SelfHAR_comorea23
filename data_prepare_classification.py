
from scipy.signal import medfilt as filter
import scipy as sp
import scipy.fftpack
import numpy as np
from sklearn.cluster import KMeans
import pandas as pd
import seaborn as sns
from matplotlib.legend_handler import HandlerPatch
import matplotlib.patches as mpatches
from glob import glob
from sklearn.metrics import f1_score
from sklearn.metrics import accuracy_score
import pickle
import pickle

def distribution_plot(laying, sitting, walking, running):
    import matplotlib.pyplot as plotter

    # The slice names of a population distribution pie chart
    pieLabels = 'Lying', 'Sitting', 'Running', 'Walking',
    Sum = np.sum([laying, sitting, running, walking])

    # Population data
    populationShare = [laying/Sum, sitting/Sum, running/Sum, walking/Sum]
    figureObject, axesObject = plotter.subplots()

    explodeTuple = (0.0, 0.1, 0.0, 0.0)

    # Draw the pie chart
    axesObject.pie(populationShare, explode=explodeTuple,
                   labels=pieLabels,
                   shadow=True,
                   autopct='%1.1f%%',
                   startangle=90)

    # Aspect ratio - equal means pie is a circle
    axesObject.axis('equal')

    plotter.show()
    return

def PKL(Users, title, dir, all):
    df = {}
    df['name'] = f"DatenPilot{title}"
    df['dataset_home_page'] = ""
    df["source_url"] = ""
    df['file_name'] = 'DatenPilot.zip'
    df['default_folder_path'] = dir
    df['save_file_name'] = 'DatenPilot.pkl'
    #df['label_list'] = ['Inactive', 'Active']
    df['label_list'] = ['laying', 'sitting', 'walking', 'running']
    #df['label_list_full_name'] = ['Inactive', 'Active']
    df['label_list_full_name'] = ['laying', 'sitting', 'walking', 'running']
    df['has_null_class'] = 'False'
    df['sampling_rate'] = 13
    df['unit'] = 1
    df['user_split'] = {}

    laying = 0
    sitting = 0
    walking = 0
    running = 0
    for i in Users:
        df['user_split'][i] = []
        for j in all:
            if i in j:
                if title =="labeled":
                    Data = pd.read_csv(df['default_folder_path'] + '/' + j + '.csv').drop(['Unnamed: 0'], axis=1)
                    #Data = pd.read_csv(df['default_folder_path'] + '/' + j + '.csv')
                    #Data = Data.loc[Data['label'] == Data['cluster label']].reset_index(drop=True)
                    #Data = Data.drop(columns=["label", "cluster", "code"])
                    #Data = Data.drop(columns=["label", "cluster"])
                    Data.columns = ["time", "acc_x", "acc_y", "acc_z", "label"]
                    Data.loc[Data['label'] == 'walkig', ['label']] = "walking"
                    #Data.loc[Data['label'] != 'running', ['label']] = 'Inactive'
                    #Data.loc[Data['label'] == 'running', ['label']] = 'Active'
                    df['user_split'][i].append([np.array(np.transpose([Data["acc_x"], Data["acc_y"], Data["acc_z"]])),
                                                np.array(Data['label'])])
                    print(title, Data['label'].unique())

                    laying = laying + len(Data.loc[Data['label'] == "laying"])
                    sitting = sitting + len(Data.loc[Data['label'] == "sitting"])
                    walking = walking + len(Data.loc[Data['label'] == "walking"])
                    running = running + len(Data.loc[Data['label'] == "running"])

                else:
                    Data = pd.read_csv(df['default_folder_path'] + '/' + j + '.csv').drop(['Unnamed: 0'], axis=1)
                    df['label_list'] = ['Nan']
                    # df['label_list'] = ['laying', 'sitting', 'walking', 'running']
                    df['label_list_full_name'] = ['Nan']
                    Data.columns = ["time", "acc_x", "acc_y", "acc_z"]
                    Data['label'] = 'Nan'
                    df['user_split'][i].append([np.array(np.transpose([Data["acc_x"], Data["acc_y"], Data["acc_z"]])),
                                                np.array(Data['label'])])
                    print(title)
    if title =="labeled":
        distribution_plot(laying, sitting, walking, running)
        laying = int(laying / (13*60*60))
        sitting = int(sitting / (13*60*60))
        walking = int(walking / (13*60*60))
        running = int(running / (13*60*60))
        T = f"laying =  {str(laying)} \n" \
            f"sitting = {str(sitting)} \n" \
            f"walking = {str(walking)} \n" \
            f"running = {str(running)} \n"

        with open(f'/home/mkfari/Project/Labeled dataset/decompressed/Code/Clustering/clustered data/activities time.txt', 'w') as f:
            f.write(T)

    # write python dict to a file
    output = open(f'/home/mkfari/Project/SelfHAR-main/run/processed_datasets/DatenPilot{title}.pkl', 'wb')
    pickle.dump(df, output)
    output.close()


labeledfolder = '/home/mkfari/Project/Labeled dataset/decompressed/Labeled Data'
#labeledfolder = "/home/mkfari/Project/Labeled dataset/decompressed/Code/Clustering/clustered data/2 stage 6 clusters filtered"
unlabeledfolder = '/home/mkfari/Project/New Dataset/uncorrupted data'
# folder = '/home/mkfari/Project/Labeled dataset/decompressed/Labeled Data/P-81f5_Tag1.csv'

LabeledDatasetFiles = sorted(glob(labeledfolder + "/*.csv"))  # Read names of dataset folders
LabeledUsersDays = [LabeledDatasetFiles[j].replace(labeledfolder + "/", "")[0:LabeledDatasetFiles[j].replace(labeledfolder + "/", "").find(".")] for j in
       range(0, len(LabeledDatasetFiles))]
LabeledUsers = [LabeledDatasetFiles[j].replace(labeledfolder + "/", "")[0:LabeledDatasetFiles[j].replace(labeledfolder + "/", "").find("_")] for j in
         range(0, len(LabeledDatasetFiles))]
LabeledUsers = np.unique(LabeledUsers)


UnlabeledDatasetFiles = sorted(glob(unlabeledfolder + "/*.csv"))  # Read names of dataset folders
UnlabeledUsersDays = [UnlabeledDatasetFiles[j].replace(unlabeledfolder + "/", "")[0:UnlabeledDatasetFiles[j].replace(unlabeledfolder + "/", "").find(".")] for j in
       range(0, len(UnlabeledDatasetFiles))]
UnlabeledUsers = [UnlabeledDatasetFiles[j].replace(unlabeledfolder + "/", "")[0:UnlabeledDatasetFiles[j].replace(unlabeledfolder + "/", "").find("_")] for j in
                  range(0, len(UnlabeledDatasetFiles))]
UnlabeledUsers = np.unique(UnlabeledUsers)

a = range(0, len(LabeledUsers), 5)
labeled = LabeledUsers
unlabeled = UnlabeledUsers
PKL(labeled, 'labeled', labeledfolder, LabeledUsersDays)
PKL(unlabeled, 'unlabeled', unlabeledfolder, UnlabeledUsersDays)
print('End')