import pandas as pd
import glob
import numpy as np
import pandas as pd
import SimpleITK as sitk
import matplotlib.pyplot as plt
import pdb

preds = glob.glob("/home/msmith/kaggle/lung/preprocessedData/*/sliced/predictedMass.bin")
preds.sort()
labels = pd.read_csv("stage1_labels.csv")
df = []
for pred in preds:
    img = np.fromfile(pred)
    patient = pred.split("/")[-3]
    label = labels[labels.id.str.contains(patient)]
    try: 
        y = label.cancer.values[0]
        df.append([patient,img.sum(),y])
    except IndexError:
        print("Test example")

df = pd.DataFrame(df)
df.columns = ["patient","mass","label"]
df.sort_values(["mass"])
df.to_csv("analysis.csv",index=0)
bins = 20 
plt.hist(df.mass[df.label==0],bins,alpha=0.5,label="0")
plt.hist(df.mass[df.label==1],bins,alpha=0.5,label="1")
plt.legend(loc="upper right")
plt.show()
