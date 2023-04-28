from glob import glob
import os
from zipfile import ZipFile


"Read folders names"

dataset_folder = "/home/mkfari/Project/Labeled dataset/Daten Pilot_innen/"
#dataset_folder = "/Users/Mohammad/OneDrive/Desktop/Mechatronics MS/Uni/Sixth semester/Master Thesis/Labeled dataset/Daten Pilot_innen/"
unzip_folder = "/home/mkfari/Project/Labeled dataset/Unzip/"
#unzip_folder = "/Users/Mohammad/OneDrive/Desktop/Mechatronics MS/Uni/Sixth semester/Master Thesis/Labeled dataset/Unzip/"

folders = [x[1] for x in os.walk(dataset_folder)][0]

activityFilesOrdered = []
stepsAndMinutesOrdered = []
for folder in folders:
    #folder = "A0015"
    #print("folder = ", folder)
    activityFiles = glob(dataset_folder + folder + "/" + "*.zip") # Read zip files names
    activityFiles = sorted(activityFiles) # make the file in order
    activityFilesOrdered.append(activityFiles) # Remove no need      activityFilesOrdered = activityFiles[1:50]
    day = ""
    filesOfOneDay = []
    for files in activityFiles:
        try:
            with ZipFile(files, 'r') as zipObj:
                # Extract all the contents of zip file in different directory
                zipObj.extractall(unzip_folder + folder + files[len(dataset_folder + folder):-4:1])

                print(unzip_folder + folder + files[len(dataset_folder + folder):-4:1])

        except:
            pass



