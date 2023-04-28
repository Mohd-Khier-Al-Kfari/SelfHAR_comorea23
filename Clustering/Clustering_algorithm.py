# -*- coding: utf-8 -*-
"""
Created on Wed May 11 13:15:00 2022

@author: Mohammad
"""
# -*- coding: utf-8 -*-
"""
Created on Wed May 11 13:15:00 2022

@author: Mohammad
"""
import matplotlib.colors as mcolors
import random
import sys
from sklearn.mixture import GaussianMixture
from sklearn.cluster import KMeans
from scipy.signal import medfilt as filter
import pandas as pd
from matplotlib.legend_handler import HandlerPatch
import matplotlib.patches as mpatches
import zipfile
from glob import glob
from sklearn.metrics import f1_score
from sklearn.metrics import accuracy_score
from sklearn.neighbors import NearestCentroid
from datetime import timedelta
from numpy import unique
from numpy import where
from matplotlib import pyplot
import numpy as np
from scipy.signal import argrelextrema
from sklearn.cluster import Birch
import datetime as dt
from scipy.stats import entropy
from scipy.signal import find_peaks
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import os
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
import scipy as sp
import scipy.fftpack
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix
from sklearn import preprocessing
import matplotlib.pyplot as plt
from scipy.spatial import ConvexHull, convex_hull_plot_2d
import joblib

"Labeling"
def Labeling(WindowSize, clusters, Data, l="cluster"):
    """
    j = 0
    for i in range(0, len(clusters)):
        Data.loc[j:j + WindowSize, l] = clusters[i]
        j = j + WindowSize
    Data.loc[j:len(Data) - 1, l] = clusters[i]
    """
    x = np.array([], dtype=np.int64).reshape(len(clusters), 0)
    for w in range(WindowSize):
        x = np.column_stack((x, clusters))
    x = x.reshape(-1)
    y = np.full((len(Data) - len(x), 1), x[-1]).reshape(-1)
    x = np.concatenate((x, y))
    Data[l] = x

    return Data


"data, classes, and clusters visualization "
def plot_sensordata_and_labels(user=None, sensordata=None, columns=None, predictions_path='', datasetname='', axis=[], F=13):
    import numpy as np
    import matplotlib.patches as mpatches
    import matplotlib.pyplot as plt
    from matplotlib.colors import LinearSegmentedColormap
    from matplotlib.ticker import StrMethodFormatter
    from plotly.subplots import make_subplots
    import plotly.graph_objects as go
    import plotly.offline as ply
    import plotly.express as px
    import plotly.subplots as sp
    from sklearn import preprocessing

    # sensordata = sensordata[90000:1000008]
    acc_x = []
    acc_y = []
    acc_z = []
    mag = []
    subjects = []
    labels = []
    if sensordata is None:
        print("input data is empty, exiting the program.")
        exit(0)

    l1 = 0
    l2 = 0
    step = F*60*60*4
    for user in sensordata["user"].unique():
        user_data = sensordata.loc[sensordata["user"] == user]
        j = 0
        for T in range(0, len(user_data), step):
            if T == range(0, len(user_data), step)[-1]:
                data = user_data[T:len(user_data)]
            else:
                data = user_data[T:T + step]
            if "time" in data:
                data = data[["user", "time", axis[0], axis[1], axis[2], "label", "cluster", "mag"]]
            else:
                #data.insert(1, 'time', range(len(data)))
                data.insert(1, 'time', data.index)
                data = data[["user", "time", axis[0], axis[1], axis[2], "label", "cluster", "mag"]]
            data.columns = ["user", 'time', 'acc_x', 'acc_y', 'acc_z', 'label', 'cluster', "mag"]

            if isinstance(data, pd.DataFrame):
                acc_x = data["acc_x"].to_numpy()
                acc_y = data["acc_y"].to_numpy()
                acc_z = data["acc_z"].to_numpy()
                mag = data["mag"].to_numpy()
                labels = data["label"].to_numpy()
                clusters = data["cluster"].to_numpy()
                # subjects = sensordata["subject"].to_numpy()

            n_classes = len(np.unique(labels))
            n_clusters = len(np.unique(clusters))
            le = preprocessing.LabelEncoder()
            le.fit(sensordata['label'])
            list(le.classes_)
            #data['code'] = 1
            data.insert(7, 'code', 1)
            l2 = l2+len(acc_x)
            figure1 = px.line(data, x='time', y=axis, labels=dict(x="Acc Data"))
            figure2 = px.area(x=data['time'], y=data['code'], color=data['label'], labels=dict(x="Classes"))
            figure3 = px.area(x=data['time'], y=data['code'], color=data['cluster'], labels=dict(x="Clusters"))
            figure4 = px.line(x=data['time'], y=mag, color=["Magnitude"] * len(mag), labels=dict(x="Magnitude Data"))
            l1 = l2
            # figure1 = px.line(x=range(len(sensordata['time'])), y=[acc_x, acc_y, acc_z])
            # figure2 = px.imshow([labels_onehot], labels=dict(x="Classes"))
            # figure3 = px.imshow([clusters_onehot], labels=dict(x="Clusters"))
            figure1_traces = []
            figure2_traces = []
            figure3_traces = []
            figure4_traces = []
            for trace in range(len(figure1["data"])):
                figure1_traces.append(figure1["data"][trace])
            for trace in range(len(figure2["data"])):
                figure2_traces.append(figure2["data"][trace])
            for trace in range(len(figure3["data"])):
                figure3_traces.append(figure3["data"][trace])
            for trace in range(len(figure4["data"])):
                figure4_traces.append(figure4["data"][trace])

            this_figure = sp.make_subplots(rows=4, cols=1, shared_xaxes=True,
                                           vertical_spacing=0.02)
            for traces in figure1_traces:
                this_figure.append_trace(traces, row=1, col=1)
            for traces in figure2_traces:
                this_figure.append_trace(traces, row=2, col=1)
            for traces in figure3_traces:
                this_figure.append_trace(traces, row=3, col=1)
            for traces in figure4_traces:
                this_figure.append_trace(traces, row=4, col=1)
            this_figure.update_layout(height=850, width=1500,
                                      title_text=user)
            path = "/home/mkfari/Project/plots/" + datasetname
            isExist = os.path.exists(path)
            if not isExist:
                os.makedirs(path)
                print("The new directory is created!")

            ply.plot(this_figure, filename=f"{path}/{user}_{j}.html")
            j = j + 1

"Features visualization"
def plot_features(title, targets, pca_features, center, xlim, ylim, name, datasets_name, activity_group, SelfHAR=np.array([])):
    fig = plt.figure(figsize=(8, 8))
    ax = fig.add_subplot(1, 1, 1)
    ax.set_xlabel('Principal Component 1', fontsize=15)
    ax.set_ylabel('Principal Component 2', fontsize=15)
    ax.set_title(title, fontsize=20)
    ax.grid()
    if len(targets) > 4:
        colors = mcolors._colors_full_map  # dictionary of all colors
        l = list(colors.items())
        random.Random(5).shuffle(l)
        colors = dict(l)
    else:
        colors = ['red', "blue", 'green', 'yellow']

    targets.sort()
    targets = np.sort(targets)
    targets = targets.tolist()
    if "Vigorous" in targets:
        colors = ["blue", 'red', 'green', 'yellow']
    elif "running" in targets:
        colors = ['red', 'green', 'yellow', "blue"]
        if "wisdm" in title:
            colors = ['green', 'yellow', "blue"]

    for target, color in zip(targets, colors):
        indicesToKeep = activity_group == target
        indicesToKeep = indicesToKeep.reshape(-1)
        ax.scatter(pca_features[indicesToKeep, 0]
                   , pca_features[indicesToKeep, 1]
                   , c=color, s=5)

    colors = ['black', "silver", 'cyan', 'brown']
    if np.any(SelfHAR):
        activity_group_SelfHAR = SelfHAR
        targets_SelfHAR = np.unique(SelfHAR)
        for target, color in zip(targets_SelfHAR, colors):
            indicesToKeep = activity_group_SelfHAR == target
            indicesToKeep = indicesToKeep.reshape(-1)
            hull = ConvexHull(pca_features[indicesToKeep])
            for simplex in hull.simplices:
                ax.plot(pca_features[indicesToKeep][simplex, 0], pca_features[indicesToKeep][simplex, 1], c=color)
        targets = targets + list(targets_SelfHAR)

    if "Clusters" in title:
        colors = ['black', 'orange', 'yellow', "white"]
        j = 3
        for i in range(0, len(datasets_name)):
            ax.scatter(center[0, i * 3:j], center[1, i * 3:j], c=colors[i], s=50, marker="v")
            j = j + 3
            # ax.scatter(center[:, 0], center[:, 1], c='black', s=50)
        ax.legend(targets + datasets_name)
    else:
        ax.legend(targets)

    targets.sort()
    targets = np.sort(targets)
    colors = ["blue", 'red', 'green', 'yellow']
    if "Clusters" in title:
        #targets = ['Active', 'Sedentary', 'Vigorous']
        for target, color in zip(targets, colors):
            indicesToKeep = activity_group == target
            indicesToKeep = indicesToKeep.reshape(-1)
            hull = ConvexHull(pca_features[indicesToKeep])
            for simplex in hull.simplices:
                ax.plot(pca_features[indicesToKeep][simplex, 0], pca_features[indicesToKeep][simplex, 1], color)
    ax.set_xlim(xlim)
    ax.set_ylim(ylim)
    fig.savefig(name)
    fig.show()

"Histogram visualization"
def plot_histogram(title, targets, features, center, xlim, ylim, name, datasets_name, activity_group):

    isExist = os.path.exists(f"/home/mkfari/Project/Data distrpution/{training_set}/")
    if not isExist:
        os.makedirs(f"/home/mkfari/Project/Data distrpution/{training_set}/")

    pca = PCA(n_components=1)
    feature_1d = pca.fit_transform(features)

    if len(targets) > 4:
        colors = mcolors._colors_full_map  # dictionary of all colors
        l = list(colors.items())
        random.Random(5).shuffle(l)
        colors = dict(l)
    else:
        colors = ['red', "blue", 'green', 'yellow']

    fig = plt.figure(figsize=(8, 8))
    ax = fig.add_subplot(1, 1, 1)
    ax.set_xlabel('Principal Component 1', fontsize=15)
    ax.set_ylabel('Probability Percentage', fontsize=15)
    ax.set_title(f"{title} distribution", fontsize=20)
    ax.grid()
    ax.set_xlim([-2, 3.5])
    j = 0
    targets = np.sort(targets)
    #targets_percentage = targets.copy()
    if "Vigorous" in targets:
        colors = ["blue", 'red', 'green', 'yellow']
    elif "running" in targets:
        colors = ['red', 'green', 'yellow', "blue"]
        if "wisdm" in title:
            colors = ['green', 'yellow', "blue"]

    targets_percentage = []
    for target, color in zip(targets, colors):
        indicesToKeep = activity_group == target
        indicesToKeep = indicesToKeep.reshape(-1)
        weight = np.ones(len(feature_1d[indicesToKeep])) / len(feature_1d)
        #targets_percentage[j] = f"{targets[j]} {str(np.round(np.sum(weight)*100,2))} %"
        targets_percentage.append(f"{targets[j]} {str(np.round(np.sum(weight) * 100, 2))} %")
        ax.hist(feature_1d[indicesToKeep], 10, color=color, histtype='step', fill=False, weights=weight)
        j = j + 1
    plt.legend(targets_percentage)
    plt.grid(True)
    plt.show()
    name = f"Histogram {title}"
    name = name.replace(" ", "_")
    fig.savefig(f"/home/mkfari/Project/Data distrpution/{training_set}/{name}")

    fig = plt.figure(figsize=(8, 8))
    ax = fig.add_subplot(1, 1, 1)
    ax.set_xlabel('Principal Component 1', fontsize=15)
    ax.set_ylabel('Probability Percentage', fontsize=15)
    ax.set_title(f"{datasets_name} distribution", fontsize=20)
    ax.grid()
    ax.set_xlim([-2, 3.5])
    #ax.set_ylim([0, 1])
    plt.hist(feature_1d, 30, facecolor="g", weights=np.ones(len(feature_1d)) / len(feature_1d))
    plt.grid(True)
    name = f"Histogram {datasets_name}"
    name = name.replace(" ", "_")
    fig.savefig(f"/home/mkfari/Project/Data distrpution/{training_set}/{name}")
    plt.show()

"Perfromance"
def performance(ture, pred):
    # print(len(DataBase["cluster"]))
    """
    ture.loc[ture == 'laying'] = 0
    ture.loc[ture == 'sitting'] = 1
    ture.loc[ture == 'walking'] = 2
    ture.loc[ture == 'running'] = 3

    pred.loc[pred == 'laying'] = 0
    pred.loc[pred == 'sitting'] = 1
    pred.loc[pred == 'walking'] = 2
    pred.loc[pred == 'running'] = 3

    ture = ture.tolist()
    pred = pred.tolist()
    """

    F1 = f1_score(ture, pred, average='weighted')

    print("F1_score weighted = ", F1)
    return F1

"Remove the mean (dc offset)"
def RemoveDC(signal):
    return (signal.T - np.mean(signal, axis=1)).T

"Mean"
def Mean(signal):
    return np.mean(signal, axis=1)

"Median"
def Median(signal):
    return np.median(signal, axis=1)

"RMS"
def Rms(signal):
    return np.sqrt(np.mean(np.square(signal), axis=1))

"Variance"
def Var(signal):
    return np.var(signal, axis=1)

"STD"
def Std(signal):
    return np.std(signal, axis=1)

"PCA"
def Pca(signal, n_components=1):
    pca = PCA(n_components=n_components)
    return pca.fit_transform(signal.T)

"Peak to peak"
def Ptp(signal):
    return np.ptp(signal, axis=1).view()

"Sum of piont"
def Sum(signal):
    return np.sum(signal, axis=1)

"Absolute Sum"
def Sa(signal):
    return np.sum(np.abs(signal), axis=1)

"total sum of squares"
def TSS(signal):
    return np.sum(np.square(signal), axis=1)

"sum of local max"
def Lmax(signal):
    lmax = []
    for i in range(np.shape(signal)[0]):
        lmax.append(np.sum(signal[i, :][argrelextrema(signal[i, :], np.greater)[0]]))
    return lmax

"sum of local min"
def Lmin(signal):
    lmin = []
    for i in range(np.shape(signal)[0]):
        lmin.append(np.sum(signal[i, :][argrelextrema(signal[i, :], np.less)[0]]))
    return lmin

"Create the spectra using fft"
def FFT(signal):
    freq = sp.fftpack.fftfreq(WindowSize) #  the length of the new window.
    # Because frequency domain is symmetrical, take only positive frequencies
    i = freq > 0
    spectra = np.abs(sp.fftpack.fft(signal))[i][0:5,:].reshape(-1)
    return spectra

"Data Normalization between 0 and 1"
def Normalizer(DataBase, axis):
    min_max_scaler = preprocessing.MinMaxScaler()
    DataBase[axis] = min_max_scaler.fit_transform(DataBase[axis])
    #from sklearn.preprocessing import StandardScaler
    #scaler = StandardScaler()
    #DataBase[axis] = scaler.fit_transform(DataBase[axis])
    return DataBase

"Features"
def get_features(sample):
    feature = Mean(sample)
    feature = np.column_stack((feature, Median(sample)))
    feature = np.column_stack((feature, Std(sample)))
    feature = np.column_stack((feature, Var(sample)))
    feature = np.column_stack((feature, Lmin(sample)))
    feature = np.column_stack((feature, Lmax(sample)))
    # feature = np.append((feature, Pca(sample)))
    feature = np.column_stack((feature, Ptp(sample)))
    feature = np.column_stack((feature, Sum(sample)))
    feature = np.column_stack((feature, Sa(sample)))
    feature = np.column_stack((feature, TSS(sample)))
    feature = np.column_stack((feature, Rms(sample)))
    # feature = np.append((feature, FFT(sample)))
    return feature

def Features(WindowSize, StepSize, data, axis, Active=True, magnitude="magnitude", axises="axises", dc="without dc", magnitude_normalize="magnitude normalize"):

    delta = timedelta(seconds=WindowSize / 2)
    #SingleAxis = np.sum(np.abs(data[axis]), axis=1) # abs sum
    #mag = np.sqrt(np.sum(np.square(data[axis]), axis=1)) #magnitude
    mag = np.array([np.sqrt(np.sum(np.square(data[axis[i:i + 3]]), axis=1)) for i in range(0, len(axis), 3)]).T  # magnitude
    if magnitude_normalize == "magnitude normalize":
        min_max_scaler = preprocessing.MinMaxScaler()
        mag = min_max_scaler.fit_transform(mag)

    mag_window = np.reshape(mag[0: int(len(mag) / WindowSize) * WindowSize], (-1, WindowSize))
    for col in data.columns:
        locals()[col] = data[col].to_numpy()
        locals()[col] = np.reshape(locals()[col][0: int(len(locals()[col]) / WindowSize) * WindowSize],
                                   (-1, WindowSize))
        if col == "label" or col == "user":
            locals()[col] = locals()[col][:, WindowSize - 1]
        else:
            locals()[col] = locals()[col].astype(float)
    features = np.array([], dtype=np.int64).reshape(np.shape(mag_window)[0], 0)

    if dc == "with dc":
        if magnitude == "magnitude":
            feature = get_features(mag_window)
            features = np.column_stack((features, feature))
        if axises == "axises":
            feature = get_features(locals()["acc_x"])
            features = np.column_stack((features, feature))
            feature = get_features(locals()["acc_y"])
            features = np.column_stack((features, feature))
            feature = get_features(locals()["acc_z"])
            features = np.column_stack((features, feature))

    if dc == "without dc":
        if magnitude == "magnitude":
            feature = get_features(RemoveDC(mag_window))
            features = np.column_stack((features, feature))
        if axises == "axises":
            feature = get_features(RemoveDC(locals()["acc_x"]))
            features = np.column_stack((features, feature))
            feature = get_features(RemoveDC(locals()["acc_y"]))
            features = np.column_stack((features, feature))
            feature = get_features(RemoveDC(locals()["acc_z"]))
            features = np.column_stack((features, feature))

    if dc == "with and without dc":
        if magnitude == "magnitude":
            feature = get_features(mag_window)
            features = np.column_stack((features, feature))

            feature = get_features(RemoveDC(mag_window))
            features = np.column_stack((features, feature))
        if axises == "axises":

            feature = get_features(locals()["acc_x"])
            features = np.column_stack((features, feature))
            feature = get_features(locals()["acc_y"])
            features = np.column_stack((features, feature))
            feature = get_features(locals()["acc_z"])
            features = np.column_stack((features, feature))

            feature = get_features(RemoveDC(locals()["acc_x"]))
            features = np.column_stack((features, feature))
            feature = get_features(RemoveDC(locals()["acc_y"]))
            features = np.column_stack((features, feature))
            feature = get_features(RemoveDC(locals()["acc_z"]))
            features = np.column_stack((features, feature))

    """  
    for i in range(0, len(data) - WindowSize + 1, StepSize):
        sample = data[axis][i:i + WindowSize].to_numpy()
        #magsample = mag[i:i + WindowSize].to_numpy().reshape((len(mag[i:i + WindowSize]),1))
        magsample = mag[i:i + WindowSize]
        sample = np.array(magsample)
        #sample = np.column_stack((sample, magsample))
        sample = np.column_stack((sample, RemoveDC(sample)))
        #sample = RemoveDC(sample)
        feature = np.array([])


        features.append(feature)
        #feature = feature.reshape((len(feature), 1))

        labels.append(data["label"][i + WindowSize - 1])
        users.append(data["user"][i + WindowSize - 1])
    """
    users = np.array(locals()["user"])
    labels = np.array(locals()["label"])
    features = np.array(features)
    data["mag"] = mag
    return features, labels, users, data

"Clustering"
def Clustering(feat, model = 0, training = False, n_clusters=3, threshold=0.1, branching_factor=100, algorithm="Kmeans"):
    if training:
        if algorithm == "Kmeans":
            model = KMeans(n_clusters=n_clusters, random_state=0)
        elif algorithm == "Birch":
            model = Birch(threshold=threshold, n_clusters=n_clusters, branching_factor=branching_factor)
        elif algorithm == "GaussianMixture":
            model = GaussianMixture(n_components=n_clusters)

        model.fit(feat)
        return model
    else:
        clusters = model.predict(feat)
        return clusters

"Max Filtering"
def Filtering(data, step , label):
    for i in range(0, 24 * 60, step):
        Max = (np.max(data.loc[(data['time'] >= data['time'][0] + pd.Timedelta(minutes=i)) & (
                    data['time'] <= data['time'][0] + pd.Timedelta(minutes=i + step)), label]))
        data.loc[(data['time'] >= data['time'][0] + pd.Timedelta(minutes=i)) & (
                    data['time'] <= data['time'][0] + pd.Timedelta(minutes=i + step)), label] = Max

    return data

"set time as index"
def set_time_as_index(dataset):
    dataset['time'] = dataset['time'].dt.to_pydatetime()
    dataset = dataset.set_index('time')
    dataset = dataset.sort_index()
    return dataset
"Downsampling"
def Downsampling(dataset , f_new, f_old):
    dataset = set_time_as_index(dataset)
    freq_new_ms = round(((1 / f_new) * 1000))
    # d = dataset[['acc_x', 'acc_y', 'acc_z']].resample(str(freq_new_ms) + "ms", axis=0).mean().dropna()
    dataset = dataset.groupby(pd.Grouper(freq=str(freq_new_ms) + "ms")).agg(
        {'acc_x': 'mean', 'acc_y': 'mean', 'acc_z': 'mean', 'user': "max", 'label': "max"}).dropna()
    return dataset

"Read dataset"
def WildData(location, DatasetName): # read the dataset and extract features for first stage
    "Read dataset"
    DatasetFiles = sorted(glob(location + "/*.csv"))  # Read names of dataset locations
    all = [DatasetFiles[j].replace(location + "/", "")[0:DatasetFiles[j].replace(location + "/", "").find(".")] for j in
           range(0, len(DatasetFiles))]
    users = [DatasetFiles[j].replace(location + "/", "")[0:DatasetFiles[j].replace(location + "/", "").find("_")] for j
             in
             range(0, len(DatasetFiles))]
    #all = ["P-2c09_8af9", "P-83b6", "P-17e8", "P-3081", "P-c0be", "P-3582"]
    #all = ["P-2c09_8af9_Tag3", "P-2c09_8af9_Tag2", "P-17e8_Tag7"]

    users = np.unique(users)
    #print("Read dataset")
    print("Users = ", users)

    DataBase = pd.DataFrame()
    for user in all:
        matching = [s for s in DatasetFiles if user in s]
        try:
            Data = pd.read_csv(matching[0], comment=';')
        except:
            continue
        Data = Data.drop(['Unnamed: 0'], axis=1)
        if DatasetName == "Elderly":
            Data["label"] = "lying"
        Data.columns = ["time", "acc_x", "acc_y", "acc_z", "label"]
        Data = Data.reset_index(drop=True)
        Data["time"] = pd.to_datetime(Data['time'])
        Data.insert(0, "user", user[0:user.find("_")])
        DataBase = pd.concat([DataBase, Data], ignore_index=True)
    DataBase.loc[DataBase["label"] == "laying", "label"] = "lying"
    DataBase.loc[DataBase["label"] == "wlking", "label"] = "walking"
    DataBase = set_time_as_index(DataBase)
    return DataBase

def Wisdm(location, F=20):
    columns = ['user', 'label', 'time', "acc_x", "acc_y", "acc_z"]
    # df = pd.read_csv(location, header = None, delim_whitespace=True, names=columns)
    data = pd.read_csv(location, header=None, delim_whitespace=True)
    data = data[0].str.split(';', 1, expand=True)
    data = data[0].str.split(',', len(columns) - 1, expand=True)
    data.columns = columns
    #data["user"] = data["user"].astype(int)
    data["user"] = data["user"].astype(str)
    data['time'] = data['time'].astype(float)
    axis = ['acc_x', 'acc_y', 'acc_z']
    data[axis] = data[axis].apply(pd.to_numeric, errors='coerce', axis=1)
    data["time"] = pd.to_datetime(data["time"], unit='ns')
    #DataBase["time"] = unix(DataBase['time'])
    #pd.to_datetime(DataBase["time"][0], unit='ns')
    data = data.dropna().reset_index(drop=True)
    if resampling is True:
        DataBase = pd.DataFrame()
        for user in data['user'].unique():
            Data = data.loc[data['user'] == user]

            Data = Downsampling(Data, f_new=13, f_old=F)
            DataBase = pd.concat([DataBase, Data])
    else:
        DataBase = data
    return DataBase

def PAMAP2(location, axis = [7,8,9], resampling = True, F=100):
    activitylist = ["lying", "sitting", "standing", "walking", "running", "cycling", "Nordic walking", "ascending stairs", "descending stairs", "vacuum cleaning", "ironing", "rope jumping"]
    DatasetFiles = sorted(glob(location + "/*.dat"))  # Read names of dataset locations
    users = [DatasetFiles[j].replace(location + "/", "")[0:DatasetFiles[j].replace(location + "/", "").find(".")] for j
             in range(0, len(DatasetFiles))]
    DataBase = pd.DataFrame()
    for user in users:
        matching = [s for s in DatasetFiles if user in s]
        Data = pd.read_csv(matching[0], header=None, delim_whitespace=True)
        Data.insert(0, "user", user)
        # DataBase = DataBase.rename(columns={2: "heart rate"})
        Data = Data.rename(columns={0: "time"})
        Data = Data.dropna(subset=np.arange(3, 53 + 1, 1)).reset_index(drop=True)
        Data = Data.rename(columns={1: "label"})
        Data = Data.loc[Data['label'] != 0].reset_index(drop=True)
        Data = Data.rename(columns={axis[0]: "acc_x", axis[1]: "acc_y", axis[2]: "acc_z"})
        Data["time"] = pd.to_datetime(Data["time"], unit='s')
        if resampling is True:
            Data = Downsampling(Data, f_new=13, f_old=F)
            DataBase = pd.concat([DataBase, Data])
        else:
            DataBase = pd.concat([DataBase, Data], ignore_index=True)


    DataBase.loc[DataBase['label'] == 1, "label"] = activitylist[0]
    DataBase.loc[DataBase['label'] == 2, "label"] = activitylist[1]
    DataBase.loc[DataBase['label'] == 3, "label"] = activitylist[2]
    DataBase.loc[DataBase['label'] == 4, "label"] = activitylist[3]
    DataBase.loc[DataBase['label'] == 5, "label"] = activitylist[4]
    DataBase.loc[DataBase['label'] == 6, "label"] = activitylist[5]
    DataBase.loc[DataBase['label'] == 7, "label"] = activitylist[6]
    DataBase.loc[DataBase['label'] == 12, "label"] = activitylist[7]
    DataBase.loc[DataBase['label'] == 13, "label"] = activitylist[8]
    DataBase.loc[DataBase['label'] == 16, "label"] = activitylist[9]
    DataBase.loc[DataBase['label'] == 17, "label"] = activitylist[10]
    DataBase.loc[DataBase['label'] == 24, "label"] = activitylist[11]
    return DataBase

def UOM(location, F=50):
    DatasetFiles = sorted([x[0] for x in os.walk(location) if "/data" in x[0]])
    users = [DatasetFiles[j].replace(location + "/", "")[0:DatasetFiles[j].replace(location + "/", "").find("/data")]
             for j in range(0, len(DatasetFiles))]
    DataBase = pd.DataFrame()

    for i in range(0, len(DatasetFiles)):
        ZipFiles = sorted(glob(DatasetFiles[i] + "/*.zip"))  # Read names of dataset locations
        accFiles = [ZipFiles[j] for j in range(0, len(ZipFiles)) if "acc" in ZipFiles[j] and "sqlite" not in ZipFiles[j] ]

        for acc in accFiles:
            zf = zipfile.ZipFile(acc)
            zl = zf.namelist()
            forearm = [zl[j] for j in range(0, len(zl)) if "forearm" in zl[j]]
            try:
                Data = pd.read_csv(zf.open(forearm[0]))
            except:
                continue
            Data = Data.rename(columns={'attr_time': "time", 'attr_x': "acc_x", 'attr_y': "acc_y", 'attr_z': "acc_z"})
            Data.insert(0, "user", users[i])
            forearm[0] = forearm[0].replace(forearm[0][0:forearm[0].find("_") + 1], "")
            forearm[0] = forearm[0][0:forearm[0].find("_")]
            Data.insert(1, "label", forearm[0])
            Data = Data.drop(columns="id")
            Data["time"] = pd.to_datetime(Data["time"], unit='ms')
            if resampling is True:
                Data = Downsampling(Data, f_new=13, f_old=F)
                DataBase = pd.concat([DataBase, Data])
            else:
                DataBase = pd.concat([DataBase, Data], ignore_index=True)
    return DataBase

def Data(location, axis = [18,19,20], F=25):
    activitylist = ["sitting", "standing", "lying on back", "lying on right", "ascending stairs", "descending stairs",
                    "standing in an elevator still", "moving around in an elevator", "walking in a parking lot",
                    "walking on a treadmill with a speed of 4 km/h", "walking on a treadmill with a speed of 4 km/h",
                    "running on a treadmill with a speed of 8 km/h", "exercising on a stepper", "exercising on a cross trainer",
                    "cycling on an exercise bike in horizontal positions", "cycling on an exercise bike in vertical positions",
                    "rowing", "jumping", "playing basketball"]
    DatasetFiles = sorted([x[0] for x in os.walk(location) if "/data" in x[0]])
    DataBase = pd.DataFrame()

    for DatasetFile in DatasetFiles:
        TextFiles = sorted(glob(DatasetFile + "/*.txt"))
        if not TextFiles:
            continue
        for TextFile in TextFiles:
            Data = pd.read_csv(TextFile, header=None, delim_whitespace=True)
            Data = Data[0].str.split(';', 1, expand=True)
            Data = Data[0].str.split(',', 45 - 1, expand=True)
            Data = Data.astype(float)
            user = TextFile.replace(location, "")
            user = user[user.find('p'):user.find("p") + 2]
            ac = TextFile.replace(location + "/", "")
            ac = ac[0:ac.find('/')]
            Data.insert(0, "user", user)
            Data.insert(1, "label", activitylist[int(ac[1:3])-1])

            DataBase = pd.concat([DataBase, Data], ignore_index=True)
    DataBase = DataBase.rename(columns={axis[0]: "acc_x", axis[1]: "acc_y", axis[2]: "acc_z"})
    return DataBase

def ActivitiesGroup(activity, group):
    activity[activity == 'rope jumping'] = group[2]
    activity[activity == 'running'] = group[2]
    activity[activity == 'exercising on a cross trainer'] = group[2]
    activity[activity == 'exercising on a stepper'] = group[2]
    activity[activity == 'running on a treadmill with a speed of 8 km/h'] = group[2]
    activity[activity == 'playing basketball'] = group[2]
    activity[activity == 'rowing'] = group[2]
    activity[activity == 'cycling'] = group[2]
    activity[activity == 'cycling on an exercise bike in horizontal positions'] = group[2]
    activity[activity == 'cycling on an exercise bike in vertical positions'] = group[2]
    activity[activity == 'Jogging'] = group[2]
    activity[activity == 'jumping'] = group[2]

    activity[activity == 'vacuum cleaning'] = group[1]
    activity[activity == 'climbingdown'] = group[1]
    activity[activity == 'climbingup'] = group[1]
    activity[activity == 'Nordic walking'] = group[1]
    activity[activity == 'nordic walking'] = group[1]
    activity[activity == 'walking'] = group[1]
    activity[activity == 'ascending stairs'] = group[1]
    activity[activity == 'descending stairs'] = group[1]
    activity[activity == 'Downstairs'] = group[1]
    activity[activity == 'Upstairs'] = group[1]
    activity[activity == 'Walking'] = group[1]
    activity[activity == 'walking on a treadmill with a speed of 4 km/h'] = group[1]
    activity[activity == 'walking in a parking lot'] = group[1]
    activity[activity == 'moving around in an elevator'] = group[1]

    #activity[activity == 'cycling'] = group[0]
    #activity[activity == 'vacuum cleaning'] = group[0]
    activity[activity == 'ironing'] = group[0]
    activity[activity == 'standing'] = group[0]
    activity[activity == 'standing in an elevator still'] = group[0]
    activity[activity == 'sitting'] = group[0]
    activity[activity == 'lying'] = group[0]
    activity[activity == 'lying on back'] = group[0]
    activity[activity == 'lying on right'] = group[0]
    activity[activity == 'laying'] = group[0]
    activity[activity == 'Standing'] = group[0]
    activity[activity == 'Sitting'] = group[0]
    return activity


class Dic:
    def __init__(self, features, pca_features, clusters, labels, center, users):
        self.features = features
        self.pca_features = pca_features
        self.clusters = clusters
        self.labels = labels
        self.center = center
        self.users = users


if __name__ == '__main__':

    "Select dataset"
    #DatasetName = "Student"
    # DatasetName = "elderly dataset"
    #DatasetName = "wisdm" # https://www.cis.fordham.edu/wisdm/dataset.php
    #DatasetName = "University of Mannheim" # https://www.uni-mannheim.de/dws/research/projects/activity-recognition/dataset/dataset-realworld/
    #DatasetName = "PAMAP2" # https://archive.ics.uci.edu/ml/datasets/pamap2+physical+activity+monitoring
    #DatasetName = "data" # https://archive.ics.uci.edu/ml/datasets/Daily+and+Sports+Activities
    F1 = {}
    random_forest_F1_dic = {}
    #DatasetNameList = ["University of Mannheim", "PAMAP2",  "Student", "wisdm", "Elderly"]
    DatasetNameList = ["PAMAP2", "University of Mannheim", "Student", "wisdm", "Elderly"]
    #DatasetNameList = ["Student", "PAMAP2", "University of Mannheim",  "wisdm", "Elderly"]
    for DatasetName in DatasetNameList:
        #DatasetName = "Elderly"
        print("Select dataset")
        print(DatasetName)

        "Read dataset"
        resampling = True
        #resampling = False
        print(f"resampling = {resampling}")
        if DatasetName == "Student":
            location = '/home/mkfari/Project/Labeled dataset/decompressed/Labeled Data'
            F_old = 13
            dataset = WildData(location=location, DatasetName=DatasetName)
            axis = ["acc_x", "acc_y", "acc_z"]
            activitylist = dataset["label"].unique()
            #activitylist = ["laying", "sitting", "walking", "running"]
            ActivitiesNumber = len(activitylist)

        elif DatasetName == "wisdm":
            location = "/home/mkfari/Project/open datasets/WISDM_ar_v1.1/WISDM_ar_v1.1_raw.txt"
            F_old = 20
            dataset = Wisdm(location=location, F=F_old)
            axis = ["acc_x", "acc_y", "acc_z"]
            activitylist = dataset["label"].unique()
            #activitylist = ["Standing", "Sitting", "Downstairs", "Upstairs", "Walking", "Jogging"]
            #activitylist = ["Standing", "Sitting", "Walking", "Jogging"]
            ActivitiesNumber = len(activitylist)

        elif DatasetName == "PAMAP2":
            location = "/home/mkfari/Project/open datasets/PAMAP2_Dataset/Protocol"
            F_old = 100
            #axis = np.arange(4, 53+1, 1)
            #axis = np.arange(4, 6 + 1, 1)
            #axis = np.arange(4, 19 + 1, 1)
            axis = np.arange(7, 9 + 1, 1)
            #axis1 = np.arange(4, 19 + 1, 1)
            #axis2 = np.arange(21, 36 + 1, 1)
            #axis3 = np.arange(38, 53 + 1, 1)
            #axis = np.concatenate([axis1, axis2, axis3])
            dataset = PAMAP2(location=location, axis=axis, resampling=resampling, F=F_old)
            activitylist = dataset["label"].unique()
            axis = ["acc_x", "acc_y", "acc_z"]
            ActivitiesNumber = len(activitylist)

        elif DatasetName == "University of Mannheim":
            location = "/home/mkfari/Project/open datasets/realworld2016_dataset"
            F_old = 50
            dataset = UOM(location=location, F=F_old)
            axis = ["acc_x", "acc_y", "acc_z"]
            activitylist = dataset["label"].unique()
            ActivitiesNumber = len(activitylist)

        elif DatasetName == "data":
            location = "/home/mkfari/Project/open datasets/data"
            F_old = 25
            axis = [18, 19, 20]
            dataset = Data(location=location, axis=axis, F=F_old)
            axis = ["acc_x", "acc_y", "acc_z"]
            activitylist = dataset["label"].unique()
            ActivitiesNumber = len(activitylist)

        elif DatasetName == "Elderly":
            location = '/home/mkfari/Project/New Dataset/uncorrupted data/csv'
            F_old = 13
            dataset = WildData(location=location, DatasetName=DatasetName)
            axis = ["acc_x", "acc_y", "acc_z"]
            activitylist = dataset["label"].unique()
            #activitylist = ["laying", "sitting", "walking", "running"]
            ActivitiesNumber = len(activitylist)

        print("Read dataset")

        "Initialize parameters"
        if resampling is True:
            F = 13
        else:
            F = F_old

        second = 4
        WindowSize = int(second * F)
        OverLapping = 0
        StepSize = WindowSize - OverLapping
        n_clusters = 3

        algorithm = "Kmeans"
        #algorithm = "GaussianMixture"
        dc = "with and without dc"
        #dc = "without dc"
        #dc = "with dc"
        #axises = "axises"
        axises = ""
        magnitude = "magnitude"
        #magnitude = ""
        magnitude_normalize = "magnitude normalize"
        #magnitude_normalize = ""

        #training_set = 'University of Mannheim'
        training_set = 'PAMAP2'
        #training_set = 'Student'
        print("Initialize parameters")

        "Target Activities"
        if resampling is True:
            dataset = dataset.loc[(dataset["label"] != "Nordic walking")]
        else:
            dataset = dataset.loc[(dataset["label"] != "Nordic walking")].reset_index(drop=True)
        dataset["label"] = dataset["label"].str.lower()
        dataset.loc[dataset["label"] == "jogging", "label"] = "running"
        DataBase = dataset
        activity = {}

        TargetActivities = ["lying", "sitting", "walking", "running"]
        #TargetActivities = dataset["label"].unique()
        activitylist = dataset["label"].unique()
        DataBase = pd.DataFrame()
        activities = []
        for i in range(len(TargetActivities)):
            for j in range(len(activitylist)):
                if (TargetActivities[i] in activitylist[j]):
                    #activities.append(activitylist[j])
                    if resampling is True:
                        Data = dataset.loc[(dataset["label"] == activitylist[j])]
                        DataBase = pd.concat([DataBase, Data])
                    else:
                        Data = dataset.loc[(dataset["label"] == activitylist[j])].reset_index(drop=True)
                        DataBase = pd.concat([DataBase, Data], ignore_index=True)
        DataBase = DataBase.rename_axis('time').sort_values(by=['user', 'time'])
        activities = DataBase["label"].unique()
        print(activities)
        print("Target Activities")



        "Normalize data"
        #DataBase = Normalizer(DataBase, axis)
        print("Normalize data")

        "Features Extraction"
        features, labels, users, DataBase = Features(WindowSize=WindowSize, StepSize=StepSize, data=DataBase, axis=axis, Active=True, magnitude=magnitude, axises=axises, dc=dc, magnitude_normalize=magnitude_normalize)
        print("Features Extraction")
        """
        "Features Selection"
        pca = PCA(n_components=30)
        features = pca.fit_transform(features)
        print("Features Selection")
        """
        "Features normalization"
        features = (features - features.min(axis=0)) / (features.max(axis=0) - features.min(axis=0))
        """
        from sklearn.preprocessing import MinMaxScaler
        scaler = MinMaxScaler()
        scaler.fit(features)
        features = scaler.transform(features)
        """
        print("Features normalization")

        "Save the model"
        if DatasetName == training_set:
            "Model Training"
            model = Clustering(feat=features, training=True, n_clusters=n_clusters, threshold=0.1, branching_factor=100,
                               algorithm=algorithm)
            clusters = Clustering(feat=features, model=model, training=False, threshold=0.1, branching_factor=100,
                                  algorithm=algorithm)
            print("Model Training")
            isExist = os.path.exists("Clustering model/")
            if not isExist:
                os.makedirs("Clustering model/")
            with open(f"Clustering model/{DatasetName} {algorithm} model.pkl", "wb") as f:
                joblib.dump(model, f)
                print("Save a model")
        else:
            model = joblib.load(f"Clustering model/{training_set} {algorithm} model.pkl")
            clusters = Clustering(feat=features, model=model, training=False, threshold=0.1, branching_factor=100,
                                  algorithm=algorithm)

            print("Load a model")

        "Set Cluster to Labels Based on STD Centers"
        pca = PCA(n_components=2)
        pca_features = pca.fit_transform(features)
        #min_max_scaler = preprocessing.MinMaxScaler()
        #pca_features = min_max_scaler.fit_transform(z)
        clf = NearestCentroid()
        clf.fit(pca_features, clusters)
        center = clf.centroids_
        """
        r = np.sqrt(center[:, 0] ** 2 + center[:, 1] ** 2)
        highActiveClusterNumber = np.argmax(r)
        inactivetClusterNumber = np.argmin(r)
        activeClusterNumber = np.argwhere((r != np.min(r)) & (r != np.max(r)))[0][0]
        """
        center = np.transpose(center)
        highActiveClusterNumber = np.argmax(center[0])
        inactivetClusterNumber = np.argmin(center[0])
        activeClusterNumber = np.argwhere((center[0] != np.min(center[0])) & (center[0] != np.max(center[0])))[0][0]

        print("Set Cluster to Labels Based on STD Centers")

        "Filtering"
        #minutes = 1
        #filterSize = (60 / second) * minutes
        #clusters = filter(clusters, kernel_size=int(filterSize))
        """
        from sklearn.neighbors import LocalOutlierFactor
        lof = LocalOutlierFactor(n_neighbors=20, contamination=0.1)
        x = np.asarray(features)
        x = x[clusters == 1]
        y_pred = lof.fit_predict(features)
        """
        print("Filtering")

        "Activities groups"
        ClusterNumber = [inactivetClusterNumber, activeClusterNumber, highActiveClusterNumber]
        act = labels.copy()
        act = ActivitiesGroup(activity=act,  group=ClusterNumber)
        act = act.astype(np.int32)
        print("Activities groups")

        "Set clusters into dataset according to its cluster"
        DataBase['cluster'] = "Nan"
        DataBase = Labeling(WindowSize, clusters, DataBase, "cluster")
        DataBase.loc[DataBase['cluster'] == highActiveClusterNumber, 'cluster'] = "Vigorous"
        DataBase.loc[DataBase['cluster'] == inactivetClusterNumber, 'cluster'] = "Sedentary"
        DataBase.loc[DataBase['cluster'] == activeClusterNumber, 'cluster'] = "Active"
        print("Set clusters into dataset according to its cluster")

        "Performance"
        F1[DatasetName] = performance(act, clusters)
        print("Performance")

        print("Confusion matrix")
        matrix = confusion_matrix(act, clusters)
        print(matrix)
        print("Confusion matrix")

        from sklearn import preprocessing

        le = preprocessing.LabelEncoder()
        print(le.fit_transform(labels))

        matrix = confusion_matrix(le.fit_transform(labels), clusters)

        print(matrix)
        print(le.inverse_transform(np.arange(0, len(np.unique(labels)), 1)))

        "Clusters name"
        clusters = clusters.astype("str")
        clusters[clusters == highActiveClusterNumber.astype("str")] = "Vigorous"
        clusters[clusters == activeClusterNumber.astype("str")] = "Active"
        clusters[clusters == inactivetClusterNumber.astype("str")] = "Sedentary"

        highActiveClusterNumber = "Vigorous"
        inactivetClusterNumber = "Sedentary"
        activeClusterNumber = "Active"
        print("Clusters name")

        path = f"/home/mkfari/Project/Features plot/training on {training_set}/"
        isExist = os.path.exists(path)
        if not isExist:
            os.makedirs(path)

        locals()[DatasetName] = Dic(features, pca_features, clusters, labels, center, users)

        "Histogram Visualisation"
        xlim = [-2, 3.5]
        ylim = [-1, 2.2]
        plot_histogram(title=DatasetName + ' labels ', targets=activities,
                       features=features, center=center, xlim=xlim, ylim=ylim,
                       name=f'{path} {DatasetName} {magnitude} {axises} {dc} {algorithm} {magnitude_normalize}',
                       datasets_name=DatasetName, activity_group=labels)

        plot_histogram(title=DatasetName + ' clusters ' + algorithm,
                       targets=[highActiveClusterNumber, activeClusterNumber, inactivetClusterNumber],
                       features=features, center=center, xlim=xlim, ylim=ylim,
                       name=f'{path} {DatasetName} {magnitude} {axises} {dc} {algorithm} {magnitude_normalize}',
                       datasets_name=DatasetName, activity_group=clusters)
        print("Histogram Visualisation")

        if DatasetName == "Elderly":
            group = [labels, clusters, pca_features]
            labels = labels[pca_features[:, 1] < 0.8]
            clusters = clusters[pca_features[:, 1] < 0.8]
            pca_features = pca_features[pca_features[:, 1] < 0.8]

        "Visualisation cluster"
        plot_features(title='Clusters ' + DatasetName + " " + algorithm,
                      targets=[highActiveClusterNumber, activeClusterNumber, inactivetClusterNumber],
                      pca_features=pca_features, center=center, xlim=xlim, ylim=ylim,
                      name=f'{path}_{DatasetName}_{magnitude}_{axises}_{dc}_{algorithm}_{magnitude_normalize}',
                      datasets_name=[DatasetName], activity_group=clusters)
        print("Visualisation cluster")

        "Visualisation label"
        plot_features(title='Labels ' + DatasetName, targets=activities,
                      pca_features=pca_features, center=center, xlim=xlim, ylim=ylim,
                      name=f'{path}_{DatasetName}_{magnitude}_{axises}_{dc}_label_{magnitude_normalize}',
                      datasets_name=[DatasetName], activity_group=labels)
        print("Visualisation label")

        if DatasetName == "Elderly":
            labels = group[0]
            clusters = group[1]
            pca_features =group[2]

        "Data, classes, and clusters visualization"
        #plot_sensordata_and_labels(sensordata=DataBase, axis=axis, datasetname=DatasetName)
        print("Data, classes, and clusters visualization")

        """
        "Random forest"
        random_forest_F1 = []
        for u in zip(np.unique(locals()[training_set].users)):
            clf = RandomForestClassifier()
            clf.fit(locals()[training_set].features[locals()[training_set].users != u[0], :],
                    locals()[training_set].labels[locals()[training_set].users != u[0]])
            if DatasetName == training_set:
                classes = clf.predict(features[users == u[0], :])
                random_forest_F1.append(performance(labels[users == u[0]], classes))
            else:
                classes = clf.predict(features)
                #if DatasetName == "Elderly":
                    #pca_features = pca.fit_transform(features)
                    #classes = classes[pca_features[:, 1] < 0.8]
                    #pca_features = pca_features[pca_features[:, 1] < 0.8]

                random_forest_F1.append(performance(labels, classes))
        print(f"random_forest_F1 = {np.mean(random_forest_F1)}")
        random_forest_F1_dic[DatasetName] = np.mean(random_forest_F1)
        print("Random forest")
        """
        "Activities groups"
        ClusterNumber = [inactivetClusterNumber, activeClusterNumber, highActiveClusterNumber]
        act = DataBase["label"].copy()
        act = ActivitiesGroup(activity=act,  group=ClusterNumber)
        DataBase.insert(3, 'actual cluster', act)
        print("Activities groups")

        "Save Dataset"
        if resampling is True:
            DataBase.to_csv("/home/mkfari/Project/datasets csv after clustering/" + DatasetName + ".csv")
        else:
            DataBase[['user', 'time', 'label', 'actual cluster', axis[0], axis[1], axis[2], 'cluster']].\
                to_csv("/home/mkfari/Project/datasets csv after clustering/" + DatasetName + ".csv", index=False)
        print("Save Dataset")

        "Features and Clusters list"

    with open(f'{path}/F1 weighted {magnitude} {axises} {dc} {algorithm} {magnitude_normalize}.txt', 'w') as f:
        f.write(str(F1))

    with open(f'{path}/F1 weighted random forest {magnitude} {axises} {dc} {algorithm} {magnitude_normalize}.txt', 'w') as f:
        f.write(str(random_forest_F1_dic))

    "without elderly"
    pca_features_list = np.array([], dtype=np.float32).reshape(0, 2)
    clusters_list = np.array([], dtype=np.float32).reshape(0, 1)
    labels_list = np.array([], dtype=np.float32).reshape(0, 1)
    center_list = np.array([], dtype=np.float32).reshape(2, 0)

    datasets_name = ["University of Mannheim", "PAMAP2", "Student"]
    for dataset_name in datasets_name:
        pca_features_list = np.row_stack((pca_features_list, locals()[dataset_name].pca_features))
        clusters_list = np.row_stack(
            (clusters_list, locals()[dataset_name].clusters.reshape(len(locals()[dataset_name].clusters), 1)))
        labels_list = np.row_stack(
            (labels_list, locals()[dataset_name].labels.reshape(len(locals()[dataset_name].labels), 1)))
        center_list = np.column_stack((center_list, locals()[dataset_name].center))

    plot_features(title='Clusters combine dataset ' + algorithm,
                  targets=[highActiveClusterNumber, activeClusterNumber, inactivetClusterNumber],
                  pca_features=pca_features_list, center=center_list, xlim=xlim, ylim=ylim,
                  name=f'{path} combine dataset {magnitude} {axises} {dc} {algorithm} {magnitude_normalize}',
                  datasets_name=datasets_name, activity_group=clusters_list)


    "with elderly"
    pca_features_list = np.array([], dtype=np.float32).reshape(0, 2)
    clusters_list = np.array([], dtype=np.float32).reshape(0, 1)
    labels_list = np.array([], dtype=np.float32).reshape(0, 1)
    center_list = np.array([], dtype=np.float32).reshape(2, 0)

    datasets_name = ["University of Mannheim", "PAMAP2", "Student", "Elderly"]
    for dataset_name in datasets_name:
        if dataset_name == "Elderly":
            locals()[dataset_name].labels = locals()[dataset_name].labels[locals()[dataset_name].pca_features[:, 1] < 0.8]
            locals()[dataset_name].clusters = locals()[dataset_name].clusters[locals()[dataset_name].pca_features[:, 1] < 0.8]
            locals()[dataset_name].pca_features = locals()[dataset_name].pca_features[locals()[dataset_name].pca_features[:, 1] < 0.8]

        pca_features_list = np.row_stack((pca_features_list, locals()[dataset_name].pca_features))
        clusters_list = np.row_stack(
            (clusters_list, locals()[dataset_name].clusters.reshape(len(locals()[dataset_name].clusters), 1)))
        labels_list = np.row_stack(
            (labels_list, locals()[dataset_name].labels.reshape(len(locals()[dataset_name].labels), 1)))
        center_list = np.column_stack((center_list, locals()[dataset_name].center))

    plot_features(title='Clusters combine dataset ' + algorithm,
                  targets=[highActiveClusterNumber, activeClusterNumber, inactivetClusterNumber],
                  pca_features=pca_features_list, center=center_list, xlim=xlim, ylim=ylim,
                  name=f'{path} combine dataset with elderly dataset {magnitude} {axises} {dc} {algorithm} {magnitude_normalize}',
                  datasets_name=datasets_name, activity_group=clusters_list)

    print("End")
    sys.exit()


""" number of pca components effects 
pca = PCA().fit(features)
plt.plot(np.cumsum(pca.explained_variance_ratio_))
plt.xlabel('number of components')
plt.ylabel('cumulative explained variance')
plt.show()
"""
