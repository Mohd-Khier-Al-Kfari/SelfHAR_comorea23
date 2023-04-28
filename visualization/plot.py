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
import os
"data, classes, and clusters visualization "
def plot_sensordata_and_labels(user=None, sensordata=None, columns=None, predictions_path='', datasetname='', axis=[], F=13, path=""):
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
                data = data[["user", "time", axis[0], axis[1], axis[2], "label", "cluster", "mag", "SelfHAR"]]
            else:
                #data.insert(1, 'time', range(len(data)))
                data.insert(1, 'time', data.index)
                data = data[["user", "time", axis[0], axis[1], axis[2], "label", "cluster", "mag", "SelfHAR"]]
            data.columns = ["user", 'time', 'acc_x', 'acc_y', 'acc_z', 'label', 'cluster', "mag", "SelfHAR"]
            #data["time"] = data.index
            if "Elderly" in datasetname:
                data["label"] = "Nan"

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
            figure2 = px.area(data, x='time', y='code', color='label', line_group="label", color_discrete_map={'lying': "red",
                                                                                                               'running': "green",
                                                                                                               'sitting': "yellow ",
                                                                                                               'walking': "blue"})
            figure3 = px.area(data, x='time', y='code', color='SelfHAR', line_group="SelfHAR", color_discrete_map={'lying': "red",
                                                                                                                   'running': "green",
                                                                                                                   'sitting': "yellow ",
                                                                                                                   'walking': "blue"})
            figure4 = px.area(data, x='time', y='code', color='cluster', line_group="cluster", color_discrete_map={'Sedentary': "red",
                                                                                                                   'Vigorous': "green",
                                                                                                                   'Active': "blue"})
            #figure2 = px.area(x=data['time'], y=data['code'], color=data['label'], labels=dict(x="Classes"))
            #figure3 = px.area(x=data['time'], y=data['code'], color=data['SelfHAR'], labels=dict(x="SelfHAR"))
            #figure4 = px.area(x=data['time'], y=data['code'], color=data['cluster'], labels=dict(x="Clusters"))
            #figure4 = px.line(x=data['time'], y=mag, color=["Magnitude"] * len(mag), labels=dict(x="Magnitude Data"))
            l1 = l2
            # figure1 = px.line(x=range(len(sensordata['time'])), y=[acc_x, acc_y, acc_z])
            # figure2 = px.imshow([labels_onehot], labels=dict(x="Classes"))
            # figure3 = px.imshow([clusters_onehot], labels=dict(x="Clusters"))
            figure1_traces = []
            figure2_traces = []
            figure3_traces = []
            figure4_traces = []

            figure1.for_each_trace(lambda trace: trace.update(fillcolor=trace.line.color))
            figure2.for_each_trace(lambda trace: trace.update(fillcolor=trace.line.color))
            figure3.for_each_trace(lambda trace: trace.update(fillcolor=trace.line.color))
            figure4.for_each_trace(lambda trace: trace.update(fillcolor=trace.line.color))

            for i in range(len(figure2['data'])):
                figure2['data'][i]['line']['width'] = 0
            for i in range(len(figure3['data'])):
                figure3['data'][i]['line']['width'] = 0
            for i in range(len(figure4['data'])):
                figure4['data'][i]['line']['width'] = 0

            for trace in range(len(figure1["data"])):
                figure1_traces.append(figure1["data"][trace])
            for trace in range(len(figure2["data"])):
                figure2_traces.append(figure2["data"][trace])
            for trace in range(len(figure3["data"])):
                figure3_traces.append(figure3["data"][trace])
            for trace in range(len(figure4["data"])):
                figure4_traces.append(figure4["data"][trace])


            this_figure = sp.make_subplots(rows=4, cols=1, shared_xaxes=True,
                                           vertical_spacing=0.04, subplot_titles=("Acc data", "Labels", "SelfHAR labels", "Clusters"))
            for traces in figure1_traces:
                this_figure.append_trace(traces, row=1, col=1)
            for traces in figure2_traces:
                this_figure.append_trace(traces, row=2, col=1)
            for traces in figure3_traces:
                this_figure.append_trace(traces, row=3, col=1)
            for traces in figure4_traces:
                this_figure.append_trace(traces, row=4, col=1)
            this_figure.update_layout(height=850, width=1500,
                                      title_text=f"{datasetname} {user}")
            isExist = os.path.exists(path)
            if not isExist:
                os.makedirs(path)
                print("The new directory is created!")

            ply.plot(this_figure, filename=f"{path} {user}_{j}.html")
            j = j + 1

            if j == 13 and "Elderly" in datasetname:
                break


"Read dataset"
folder = "/home/mkfari/Project/datasets csv after clustering/After SelfHAR PAMAP2"
#folder = "/home/mkfari/Project/datasets csv after clustering/After SelfHAR University of Mannheim"

DatasetFiles = sorted(glob(folder + "/*.csv"))  # Read names of dataset folders
#all = [DatasetFiles[j].replace(folder + "/", "")[0:DatasetFiles[j].replace(folder + "/", "").find(".")] for j in range(0, len(DatasetFiles))]
all = ['Elderly_1', 'Student', 'University of Mannheim', 'wisdm']
all = ['University of Mannheim', 'Student', 'Elderly_1']
#all = ['PAMAP2', 'Student', 'Elderly_1']
#users = [DatasetFiles[j].replace(folder + "/", "")[0:DatasetFiles[j].replace(folder + "/", "").find("_")] for j in range(0, len(DatasetFiles))]
#users = np.unique(users)
#users = np.roll(users, -1)
users = np.unique(all)
print(users)
for datasetname in all:
    #datasetname = "Student"
    print(datasetname)
    matching = [s for s in DatasetFiles if datasetname in s]
    #matching = DatasetFiles
    try:
        DataBase = pd.read_csv(matching[0], comment=';')
    except:
        continue
    DataBase["time"] = pd.to_datetime(DataBase['time'])
    print(np.size(DataBase))
    print(DataBase[0:10])
    axis = ["acc_x", "acc_y", "acc_z"]
    print("Read dataset")
    plot_sensordata_and_labels(sensordata=DataBase, axis=axis, datasetname=datasetname, path=f"{folder} plots/ {datasetname}/")

print("end")