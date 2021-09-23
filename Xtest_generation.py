'''
created by Oanh Thi La
purpose: Gennerate X-testing for testing model (can be used for users)
'''


import numpy as np
from numpy import load, save, concatenate
import pandas as pd
import tkinter as tkinter
from tkinter import filedialog
from sklearn import preprocessing

# Loading the TOA_angle_AOT patch (3*3)
test_paths = filedialog.askopenfilenames(title='Choose all testing files', filetypes=[("NPY", ".npy")])
xtesting = np.load(test_paths[00])
# xtesting = np.concatenate((BaraBonita_20140506, BaraBonita_20141013), axis=0) .astype('float32')

n_testsamples = xtesting.shape[0]
TOA_xtesting = np.zeros([n_testsamples, 8, 3, 3])
for i in range(0, n_testsamples):
    for j in range(0, 8):
        TOA_xtesting[i][j] = xtesting[i][j]
TOA_xtesting_reshape = TOA_xtesting.reshape(TOA_xtesting.shape[0], 3, 3, 8, 1).astype('float32')
path_save = tkinter.filedialog.asksaveasfilename(title=u'Save to TOA_Xtesting npy file', filetypes=[("NPY", ".npy")])
save(path_save, TOA_xtesting_reshape)

angles_xtesting = np.zeros([n_testsamples, 3, 3, 3])
for m in range(0, n_testsamples):
    for k in range(0, 3):
        angles_xtesting[m][k] = xtesting[m][k + 8]
angles_xtesting_reshape = angles_xtesting.reshape(angles_xtesting.shape[0], 3 * 3 * 3).astype('float32')
## normalize angles data from (-1,1) to scale (0,1)
pdread_angles_test = pd.DataFrame(angles_xtesting_reshape)
scaler2 = preprocessing.MinMaxScaler()
colums2 = pdread_angles_test.columns
transform2 = scaler2.fit_transform(pdread_angles_test)
angles_xtesting_normalize = pd.DataFrame(transform2, columns=colums2)
angles_xtesting_normalize.head()
angles_xtesting_final = angles_xtesting_normalize.to_numpy()
path_save = tkinter.filedialog.asksaveasfilename(title=u'Save to Angles_Xtesting npy file', filetypes=[("NPY", ".npy")])
save(path_save, angles_xtesting_final)

AOT_xtesting = np.zeros([n_testsamples, 1, 3, 3])
for n in range(0, n_testsamples):
    for h in range(0, 1):
        AOT_xtesting[n][h] = xtesting[n][h + 11]
AOT_xtesting_reshape = AOT_xtesting.reshape(AOT_xtesting.shape[0], 3 * 3 * 1).astype('float32')
path_save = tkinter.filedialog.asksaveasfilename(title=u'Save to AOT_xtesting npy file', filetypes=[("NPY", ".npy")])
save(path_save, AOT_xtesting_reshape)

