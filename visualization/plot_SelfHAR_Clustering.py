import Clustering_algorithm
import pandas as pd
from glob import glob
import numpy as np
from sklearn.decomposition import PCA
from sklearn.neighbors import NearestCentroid
import os
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import random
from scipy.spatial import ConvexHull, convex_hull_plot_2d
import gc
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import seaborn as sns
from sklearn import preprocessing
import sys

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
        colors = ['red', 'green', 'yellow', "blue"]

    targets.sort()
    for target, color in zip(targets, colors):
        indicesToKeep = activity_group == target
        indicesToKeep = indicesToKeep.reshape(-1)
        ax.scatter(pca_features[indicesToKeep, 0]
                   , pca_features[indicesToKeep, 1]
                   , c=color, s=5)
    """
    if "Clusters" in title:
        colors = ['black', 'orange', 'yellow', "white"]
        j = 4
        for i in range(0, len(datasets_name)):
            ax.scatter(center[0, i * 4:j], center[1, i * 4:j], c=colors[i], s=50, marker="v")
            j = j + 4
            # ax.scatter(center[:, 0], center[:, 1], c='black', s=50)
        #ax.legend(targets + list(targets_SelfHAR) + datasets_name)
    else:
        ax.legend(targets)
    """

    colors = ['black', "orange", 'cyan']
    if np.any(SelfHAR):
        activity_group_SelfHAR = SelfHAR
        targets_SelfHAR = np.unique(SelfHAR)
        for target, color in zip(targets_SelfHAR, colors):
            indicesToKeep = activity_group_SelfHAR == target
            indicesToKeep = indicesToKeep.reshape(-1)
            hull = ConvexHull(pca_features[indicesToKeep])
            for simplex in hull.simplices:
                ax.plot(pca_features[indicesToKeep][simplex, 0], pca_features[indicesToKeep][simplex, 1], c=color)
        #targets = targets + list(targets_SelfHAR)
        #leg = ax.legend(targets + datasets_name + list(targets_SelfHAR))
        leg = ax.legend(targets + list(targets_SelfHAR))
        leg.legendHandles[-2].set_color(colors[-2])
        leg.legendHandles[-1].set_color(colors[-1])

    """
    colors = ['red', "blue", 'green', 'yellow']
    if "Clusters" in title:
        #targets = ['Active', 'Sedentary', 'Vigorous']
        for target, color in zip(targets, colors):
            indicesToKeep = activity_group == target
            indicesToKeep = indicesToKeep.reshape(-1)
            hull = ConvexHull(pca_features[indicesToKeep])
            for simplex in hull.simplices:
                ax.plot(pca_features[indicesToKeep][simplex, 0], pca_features[indicesToKeep][simplex, 1], color)
    """
    ax.set_xlim(xlim)
    ax.set_ylim(ylim)
    fig.savefig(name)
    fig.show()
    gc.collect()

"Confusion matrix visualization"
def plot_confusion_matrix(y_true=pd.DataFrame(), y_pred=pd.DataFrame(), X_label='Clusters', Y_label='SelfHAR labels', Title='Confusion Matrix',name=""):

    if y_true.empty or y_pred.empty:
        print("No data")
        return 0

    le = preprocessing.LabelEncoder()
    y_true_data = le.fit_transform(y_true)
    y_true_label = le.inverse_transform(np.arange(0, len(np.unique(y_true_data)), 1))
    y_pred_data = le.fit_transform(y_pred)
    y_pred_label = le.inverse_transform(np.arange(0, len(np.unique(y_pred_data)), 1))
    matrix = confusion_matrix(y_true_data, y_pred_data)
    #matrix = matrix[:, 0:3]
    matrix = matrix / np.sum(matrix, axis=0)
    matrix = matrix[:, ~np.isnan(matrix).any(axis=0)]

    df = pd.DataFrame(matrix, columns=y_pred_label, index=y_true_label)

    plt.figure(figsize=(10, 7))
    g = sns.heatmap(df, annot=True, cmap="Blues", annot_kws={"size": 16, 'fontweight': 'bold'}, fmt=".2%")
    g.set_xticklabels(g.get_xmajorticklabels(), fontsize=14)
    g.set_yticklabels(g.get_ymajorticklabels(), fontsize=14)

    plt.xlabel(X_label, fontsize=18)
    plt.ylabel(Y_label, fontsize=18)
    plt.title(Title, fontsize=20)
    plt.savefig(name)
    plt.show()
    gc.collect()

if __name__ == "__main__":

    F = 13
    second = 4
    WindowSize = int(second * F)
    OverLapping = 0
    StepSize = WindowSize - OverLapping
    n_clusters = 3
    axis = ["acc_x", "acc_y", "acc_z"]
    algorithm = "Kmeans"
    # algorithm = "GaussianMixture"
    dc = "with and without dc"
    # dc = "without dc"
    # dc = "with dc"
    # axises = "axises"
    axises = ""
    magnitude = "magnitude"
    # magnitude = ""
    magnitude_normalize = "magnitude normalize"
    # magnitude_normalize = ""
    xlim = [-2, 3.5]
    ylim = [-1, 2.2]
    training_set = "PAMAP2"
    #training_set = "University of Mannheim"
    location = f"/home/mkfari/Project/datasets csv after clustering/After SelfHAR {training_set}/"
    DatasetFiles = sorted(glob(location + "*.csv"))  # Read names of dataset locations
    i = 0
    DataBase = pd.DataFrame()
    elderly = []
    for DatasetFile in DatasetFiles:
        #DatasetFile = '/home/mkfari/Project/datasets csv after clustering/After SelfHAR PAMAP2/University of Mannheim.csv'
        DatasetName = DatasetFile.replace(location, "").replace(".csv", "")
        if "Elderly" in DatasetName:
            #continue
            df = pd.read_csv(DatasetFile)
            DataBase = pd.concat([DataBase, df])
            print(f"DataBase = {len(DataBase)}")
            print(f"df = {len(df)}")
            #elderly.append(DatasetName)
            DatasetName = "Elderly"
            if "Elderly" in DatasetFiles[i+1]:
                i = i + 1
                continue
        else:
            DataBase = pd.read_csv(DatasetFile)
        print(DatasetName)

        "features extraction"
        features, labels, users, _ = Clustering_algorithm.Features(WindowSize=WindowSize, StepSize=StepSize, data=DataBase[['user', 'acc_x', 'acc_y', 'acc_z', 'label']],
                                                                   axis=axis, Active=True, magnitude=magnitude, axises=axises, dc=dc, magnitude_normalize=magnitude_normalize)
        print("features extraction")

        for col in ["SelfHAR", "cluster"]:
            locals()[col] = DataBase[col].to_numpy()
            locals()[col] = np.reshape(locals()[col][0: int(len(locals()[col]) / WindowSize) * WindowSize],
                                       (-1, WindowSize))
            locals()[col] = locals()[col][:, WindowSize - 1]

        features = (features - features.min(axis=0)) / (features.max(axis=0) - features.min(axis=0))

        pca = PCA(n_components=2)
        pca_features = pca.fit_transform(features)
        clf = NearestCentroid()
        clf.fit(pca_features, locals()["cluster"])
        center = clf.centroids_
        center = np.transpose(center)
        """
        plot_features(title='Clusters ' + DatasetName + " " + algorithm,
                      targets=list(DataBase["SelfHAR"].unique()), pca_features=pca_features,
                      center=center, xlim=xlim, ylim=ylim,
                      name=f'{path} {DatasetName} {magnitude} {axises} {dc} {algorithm} {magnitude_normalize}',
                      datasets_name=[DatasetName], activity_group=locals()["SelfHAR"])
        """
        """
        plot_features(title='Clusters ' + DatasetName + " " + algorithm,
                      targets=list(DataBase["cluster"].unique()), pca_features=pca_features,
                      center=center, xlim=xlim, ylim=ylim,
                      name=f'{path} {DatasetName} {magnitude} {axises} {dc} {algorithm} {magnitude_normalize}',
                      datasets_name=[DatasetName], activity_group=locals()["cluster"], SelfHAR=locals()["SelfHAR"])
        """

        clf = NearestCentroid()
        clf.fit(pca_features, locals()["SelfHAR"])
        center = clf.centroids_
        center = np.transpose(center)

        if DatasetName == "Elderly":
            group = [locals()["SelfHAR"], locals()["cluster"], pca_features]
            locals()["SelfHAR"] = locals()["SelfHAR"][pca_features[:, 1] < 0.8]
            locals()["cluster"] = locals()["cluster"][pca_features[:, 1] < 0.8]
            pca_features = pca_features[pca_features[:, 1] < 0.8]

        path = f"/home/mkfari/Project/Features plot/SelfHAR/{training_set}/"
        isExist = os.path.exists(path)
        if not isExist:
            os.makedirs(path)


        plot_features(title= DatasetName + " " + "SelfHAR",
                              targets=list(DataBase["SelfHAR"].unique()), pca_features=pca_features,
                              center=center, xlim=xlim, ylim=ylim,
                              name=f'{path}{DatasetName}_{magnitude}_{axises}_{dc}_{algorithm}_{magnitude_normalize}',
                              datasets_name=[DatasetName], activity_group=locals()["SelfHAR"], SelfHAR=locals()["cluster"])

        "plot confusion matrix"
        plot_confusion_matrix(y_true=DataBase['SelfHAR'], y_pred=DataBase["cluster"],
                              X_label="Clusters", Y_label='SelfHAR labels', Title=f'{DatasetName} Evaluation Matrix',
                              name=f'{path}Evaluation_matrix_{DatasetName}')
        print("plot confusion matrix")

        plot_confusion_matrix(y_true=DataBase['SelfHAR'], y_pred=DataBase["label"],
                              X_label="Labels", Y_label='SelfHAR labels', Title=f'{DatasetName} Confusion Matrix',
                              name=f'{path}Confusion_matrix_{DatasetName}')
        print("plot confusion matrix")

        DataBase = pd.DataFrame()

    print("End")
    sys.exit()
