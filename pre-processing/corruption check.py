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

pd.options.mode.chained_assignment = None  # default='warn'

"STD overall sensor data"


def STD(DataBase, WindowSize, StepSize, axis):
    from datetime import timedelta
    delta = timedelta(seconds=WindowSize / 2)
    DataSlot = [np.std(DataBase.loc[i:i + WindowSize][axis]) for i in range(0, len(DataBase) - WindowSize, StepSize) if
                np.absolute(DataBase['time'][i + WindowSize] - DataBase['time'][i]) < delta]

    # random.shuffle(DataSlot)
    return DataSlot
    # random.shuffle(DataSlot)
    return DataSlot


"Create the spectra using fft"


def FFT(WindowSize, StepSize, DataBase, axis):
    from datetime import timedelta
    delta = timedelta(seconds=WindowSize / 2)
    freq = sp.fftpack.fftfreq(WindowSize)  # the length of the new window.
    # Because frequency domain is symmetrical, take only positive frequencies
    i = freq > 0
    spectra = [np.abs(
        sp.fftpack.fft((DataBase[j:j + WindowSize][axis] - np.mean(DataBase[j:j + WindowSize][axis])).to_numpy()))[i]
               for j in range(0, len(DataBase) - WindowSize, StepSize) if
               np.absolute(DataBase['time'][j + WindowSize] - DataBase['time'][j]) < delta]

    return spectra


"Data Normalization between 0 and 1"


def Normalizer(DataBase, axis):
    from sklearn import preprocessing
    min_max_scaler = preprocessing.MinMaxScaler()
    DataBase[axis] = min_max_scaler.fit_transform(DataBase[axis])
    return DataBase


"K-mean Clustering"


def Clustering(Data, kmeans, ClusterNumbers=2, train=1, freqDomain=1, F=13):
    if train and freqDomain:
        nonActive = np.zeros((1, F - 1))
        nonActive[0,0] = 5
        Active = np.ones((1, F - 1))*5
        Active[0,0] = 1
        Center = np.concatenate((nonActive, Active), axis=0)
        kmeans = KMeans(n_clusters=ClusterNumbers, init=Center, max_iter=600, random_state=0, n_init=1)
        #kmeans = KMeans(n_clusters=ClusterNumbers, random_state=0)
        kmeans.fit(Data)
        return kmeans
    elif train and not freqDomain:
        Center = np.concatenate((np.zeros((1, 3)), np.ones((1, 3))), axis=0)
        kmeans = KMeans(n_clusters=ClusterNumbers,  init=Center, max_iter=600, random_state=0, n_init=1)
        #kmeans = KMeans(n_clusters=ClusterNumbers, random_state=0)
        kmeans.fit(Data)
        return kmeans
    else:
        predictions = kmeans.predict(Data)
        return predictions


"Data Labeling according to its cluster"


def Labeling(WindowSize, clusters, label):
    j = 0
    for i in range(0, len(clusters)):
        label.loc[j:j + WindowSize] = clusters[i]
        j = j + WindowSize
    return label


"data, classes, and clusters visualization "


class HandlerEllipse(HandlerPatch):

    def create_artists(self, legend, orig_handle,
                       xdescent, ydescent, width, height, fontsize, trans):
        center = 0.5 * width - 0.5 * xdescent, 0.5 * height - 0.5 * ydescent
        p = mpatches.Ellipse(xy=center, width=height + xdescent,
                             height=height + ydescent)
        self.update_prop(p, orig_handle, legend)
        p.set_transform(trans)
        return [p]


def plot_sensordata_and_labels(user=None, sensordata=None, columns=None, predictions_path='', datasetname=''):
    import numpy as np
    import matplotlib.patches as mpatches
    import matplotlib.pyplot as plt
    from matplotlib.colors import LinearSegmentedColormap
    from matplotlib.ticker import StrMethodFormatter
    # sensordata = sensordata[90000:1000008]
    acc_x = []
    acc_y = []
    acc_z = []
    subjects = []
    labels = []

    sensordata.columns = ['time', 'acc_x', 'acc_y', 'acc_z', 'cluster', 'label']
    if sensordata is None:
        print("input data is empty, exiting the program.")
        exit(0)
    if isinstance(sensordata, pd.DataFrame):
        acc_x = sensordata["acc_x"].to_numpy()
        acc_y = sensordata["acc_y"].to_numpy()
        acc_z = sensordata["acc_z"].to_numpy()
        labels = sensordata["label"].to_numpy()
        clusters = sensordata["cluster"].to_numpy()
        # subjects = sensordata["subject"].to_numpy()

    n_classes = len(np.unique(labels))
    n_clusters = len(np.unique(clusters))
    """
    # plot 1:
    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, sharex=True)
    #fig, (ax1, ax2) = plt.subplots(2, 1, sharex=True)
    ax1.plot(acc_x, color='blue')
    ax1.plot(acc_y, color='green')
    ax1.plot(acc_z, color='red')
    ax1.set_ylabel("Acceleration (mg)")
    ax1.legend(["acc x", "acc y", "acc z"])

    unordered_unique_labels, first_occurences, labels_onehot = np.unique(labels, return_inverse=True,
                                                                         return_index=True)
    order = []

    ordered_unique_onehot_labels = first_occurences.copy()
    ordered_unique_onehot_labels.sort()
    ordered_labels = []
    for index in ordered_unique_onehot_labels:
        ordered_labels.append(unordered_unique_labels[np.where(first_occurences == index)[0][0]])
        order.append(np.where(first_occurences == index)[0][0])

    unordered_unique_clusters, clusters_occurences, clusters_onehot = np.unique(clusters, return_inverse=True,
                                                                         return_index=True)
    order_clusters = []

    ordered_unique_onehot_clusters = clusters_occurences.copy()
    ordered_unique_onehot_clusters.sort()
    ordered_clusters = []
    for index in ordered_unique_onehot_clusters:
        ordered_clusters.append(unordered_unique_clusters[np.where(clusters_occurences == index)[0][0]])
        order_clusters.append(np.where(clusters_occurences == index)[0][0])

    colors1 = sns.color_palette(palette="hls", n_colors=n_classes).as_hex()
    colors2 = sns.color_palette(palette="Spectral", n_colors=n_clusters).as_hex()
    original_colors = colors1.copy()
    original_colors_cluster = colors2.copy()
    for i in range(0, n_classes):
        colors1[i] = original_colors[order[i]]

    for i in range(0, n_clusters):
        colors2[i] = original_colors_cluster[order_clusters[i]]

    cmap1 = LinearSegmentedColormap.from_list(name='My Colors1', colors=colors1, N=len(colors1))
    cmap2 = LinearSegmentedColormap.from_list(name='My Colors2', colors=colors2, N=len(colors2))

    ax2.set_yticks([])
    ax2.set_ylabel("Classes")
    ax2.pcolor([labels_onehot], cmap=cmap1)

    ax3.set_yticks([])
    ax3.set_ylabel("Clusters")
    ax3.pcolor([clusters_onehot], cmap=cmap2)

    plt.subplots_adjust(wspace=0, hspace=0)
    #plt.suptitle(datasetname)
    plt.suptitle("STD and Freq. K-mean Clustering Over 3 axis")

    c = [mpatches.Circle((0.5, 0.5), radius=0.25, facecolor=colors1[i], edgecolor="none") for i in
         range(n_classes)]

    plt.legend(c, unordered_unique_labels, bbox_to_anchor=(0.5, -0.15), loc='upper center', ncol=n_classes,
               fancybox=True, shadow=True,
               handler_map={mpatches.Circle: HandlerEllipse()}).get_frame()
    """
    """
    c2 = [mpatches.Circle((0.5, 0.5), radius=0.25, facecolor=colors2[i], edgecolor="none") for i in
         range(n_clusters)]

    plt.legend(c2, unordered_unique_clusters, bbox_to_anchor=(0.5, -0.15), loc='upper center', ncol=n_clusters,
               fancybox=True, shadow=True,
               handler_map={mpatches.Circle: HandlerEllipse()}).get_frame()

    plt.gca().xaxis.set_major_formatter(StrMethodFormatter('{x:,.0f}'))  # No decimal places
    plt.savefig("plot2/"+user + '.png')
    plt.show()
    """
    from plotly.subplots import make_subplots
    import plotly.graph_objects as go
    import plotly.offline as ply
    import plotly.express as px
    import plotly.subplots as sp

    from sklearn import preprocessing

    le = preprocessing.LabelEncoder()
    le.fit(sensordata['label'])
    list(le.classes_)
    sensordata['code'] = 1

    figure1 = px.line(x=sensordata['time'], y=[acc_x, acc_y, acc_z])
    figure2 = px.area(x=sensordata['time'], y=sensordata['code'], color=sensordata['label'], labels=dict(x="Classes"))
    figure3 = px.area(x=sensordata['time'], y=sensordata['cluster'], labels=dict(x="Clusters"))

    # figure1 = px.line(x=range(len(sensordata['time'])), y=[acc_x, acc_y, acc_z])
    # figure2 = px.imshow([labels_onehot], labels=dict(x="Classes"))
    # figure3 = px.imshow([clusters_onehot], labels=dict(x="Clusters"))
    figure1_traces = []
    figure2_traces = []
    figure3_traces = []
    for trace in range(len(figure1["data"])):
        figure1_traces.append(figure1["data"][trace])
    for trace in range(len(figure2["data"])):
        figure2_traces.append(figure2["data"][trace])
    for trace in range(len(figure3["data"])):
        figure3_traces.append(figure3["data"][trace])

    this_figure = sp.make_subplots(rows=3, cols=1, shared_xaxes=True,
                                   vertical_spacing=0.02)
    for traces in figure1_traces:
        this_figure.append_trace(traces, row=1, col=1)
    for traces in figure2_traces:
        this_figure.append_trace(traces, row=2, col=1)
    for traces in figure3_traces:
        this_figure.append_trace(traces, row=3, col=1)
    this_figure.update_layout(height=850, width=1500,
                              title_text=user)
    ply.plot(this_figure, filename="plot2/" + user + '.html')


"Perfromance"


def performance(DataBase):
    # print(len(DataBase["cluster"]))

    labels = np.unique(DataBase['label'])
    active = 'running'
    notActive = np.delete(labels, np.where(active == labels))

    DataBase.insert(loc=5, column="active", value=DataBase['label'])
    DataBase['active'] = DataBase['active'].replace(notActive, False)
    DataBase['active'] = DataBase['active'].replace(active, True)
    DataBase['cluster'] = DataBase['cluster'].replace(0, False)
    DataBase['cluster'] = DataBase['cluster'].replace(1, True)
    # print("F1_score micro = ", f1_score(DataBase['active'], DataBase['cluster'], average='micro'))
    # print("F1_score  = ", f1_score(DataBase['active'], DataBase['cluster']))
    # print("Accuracy = ", accuracy_score(DataBase['active'], DataBase['cluster']))

    # print("DataBase['cluster'] == True", sum(DataBase['cluster'] == True))
    # print("DataBase['cluster'] == False", sum(DataBase['cluster'] == False))

    F1 = f1_score(DataBase['active'], DataBase['cluster'], average='micro')
    Accuracy = accuracy_score(DataBase['active'], DataBase['cluster'])

    print("F1_score micro = ", F1)
    print("Accuracy = ", Accuracy)
    return F1, Accuracy


"Read dataset"
# folder = '/Users/Mohammad/OneDrive/Desktop/Mechatronics MS/Uni/Sixth semester/Master Thesis/dataset/decompressed/A0007.csv'
# folder = '/home/mkfari/Project/Labeled dataset/decompressed/Labeled Data/P-17e8_Tag1.csv'
#folder = '/home/mkfari/Project/Labeled dataset/decompressed/Labeled Data'
# folder = '/home/mkfari/Project/Labeled dataset/decompressed/Labeled Data/P-81f5_Tag1.csv'
folder = '/home/mkfari/Project/New Dataset/csv'

DatasetFiles = sorted(glob(folder + "/*.csv"))  # Read names of dataset folders
all = [DatasetFiles[j].replace(folder + "/", "")[0:DatasetFiles[j].replace(folder + "/", "").find(".")] for j in
       range(0, len(DatasetFiles))]
users = [DatasetFiles[j].replace(folder + "/", "")[0:DatasetFiles[j].replace(folder + "/", "").find("_")] for j in
         range(0, len(DatasetFiles))]
users = np.unique(users)
# users = np.roll(users, -1)
print(users)

"Windows parameter"
F = 13
second = 2
WindowSize = int(second * F)
# OverLapping = int(WindowSize/2)
OverLapping = 0
StepSize = WindowSize - OverLapping
print("Windows parameter")

"DataBases"
DataBase = {}
DataBaseSTD = {}
DataBaseSpectraX = {}
DataBaseSpectraY = {}
DataBaseSpectraZ = {}

#all = [all[5]]
for user in all:
    "Data reading"
    matching = [s for s in DatasetFiles if user in s]
    print(matching)
    Data = pd.read_csv(matching[0], comment=';')
    Data.columns = ["time", "acc_x", "acc_y", "acc_z"]
    Data = Data.reset_index(drop=True)
    Data["time"] = pd.to_datetime(Data['time'])
    axis = ["acc_x", "acc_y", "acc_z"]
    print("Read dataset")

    days = set(Data['time'].dt.day)
    for day in days:
        df = Data[Data['time'].dt.day == day]
        df = df.reset_index(drop=True)
        "Normalize data"
        # Data = Normalizer(Data, axis)
        print("Normalize data")

        "Apply STD for data"
        WindowData = STD(df, WindowSize, StepSize, axis)
        print("Apply STD for data")

        "Create spectra"
        spectraX = FFT(WindowSize, StepSize, df, axis=axis[0])
        spectraY = FFT(WindowSize, StepSize, df, axis=axis[1])
        spectraZ = FFT(WindowSize, StepSize, df, axis=axis[2])
        print("Create spectra")

        "Train cluster over 3 axis"
        ClusterNumbers = 2
        KX = Clustering(Data=spectraX, kmeans=0, ClusterNumbers=ClusterNumbers, train=1, freqDomain=1, F=F)
        KY = Clustering(Data=spectraY, kmeans=0, ClusterNumbers=ClusterNumbers, train=1, freqDomain=1, F=F)
        KZ = Clustering(Data=spectraZ, kmeans=0, ClusterNumbers=ClusterNumbers, train=1, freqDomain=1, F=F)
        KSTD = Clustering(Data=WindowData, kmeans=0, ClusterNumbers=ClusterNumbers, train=1, freqDomain=0, F=F)
        print("KX = ", np.mean(KX.cluster_centers_[1]))
        print("KY = ", np.mean(KY.cluster_centers_[1]))
        print("KZ = ", np.mean(KZ.cluster_centers_[1]))
        print("KSTD = ", np.mean(KSTD.cluster_centers_[1]))
        print("Train cluster over 3 axis")

        uncorrupted = np.all([np.mean(KX.cluster_centers_[1]) < 5, np.mean(KY.cluster_centers_[1]) < 5, np.mean(KZ.cluster_centers_[1]) < 5, np.mean(KSTD.cluster_centers_[1]) < 2]) and np.all([np.mean(KX.cluster_centers_[1]) > 0.1, np.mean(KY.cluster_centers_[1]) > 0.1, np.mean(KZ.cluster_centers_[1]) > 0.1, np.mean(KSTD.cluster_centers_[1]) > 0.01])
        if uncorrupted:
            df.to_csv("/home/mkfari/Project/New Dataset/uncorrupted data/" + user + ".csv")
            print("DeviceID: " + user + "_" + str(day) + " saved.")
        else:
            print(user + " is corrupted")

print("End")
import sys

sys.exit()
