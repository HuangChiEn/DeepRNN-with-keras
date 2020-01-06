# Data_Preprocessing.py
# (1) For propercessing dataSet in matlab
import numpy as np
import math
import scipy.io as scio
import time

def load_data(file_path='man', datNam='Man1'):
    print(file_path) 
    dataSet = scio.loadmat(file_path)
    return dataSet[datNam]
    
def preprocess(oriDatSet, seq_len=32, sample_num=1500, traRto=0.8): 
    q, mod = divmod(len(oriDatSet), seq_len)
    ## For confirm each time stamp can be fully padding!
    if(mod==0):
        oriDatSet = pad_sequences(oriDatSet, maxlen=len(oriDatSet)+mod, padding='post')
        
    # Splite the sequence length in each time_stamp :
    # Specification :=> for each time_stamp are given 32 seq_len, to predict 32 output
    #                   That's (Many to Many model), and each input have 2 features(PPG, ECG).
    traSetLst = list()
    tstSetLst = list()
    package = list()
    for i in range(seq_len, len(oriDatSet[0])+1, seq_len):
        package.append(oriDatSet[:, (i-seq_len):(i)])  
    traSiz = len(package)*traRto
    for i in range(len(package)):
        if i < traSiz:
            traSetLst.extend([package[i]])
        else:
            tstSetLst.extend([package[i]])
    return traSetLst, tstSetLst

def pack_time_stamp(dataLst, fea_num, pakNum=0):
    datStmp = list()
    dat2Stmp = list()
    labStmp = list()
    datTmp = list()
    for i in range(len(dataLst)): # split the raw-data and predict label
        datTmp.append(dataLst[i][0:2])   # ecg, ppg feature (raw-data) 
        dat2Stmp.append(dataLst[i][2:3])  # just 1 output-feature (label)
        
    colPack = list()
    for i in range(len(datTmp)):  # May be generalize in multi-feature with slightly modification(add loop).
        for j in range(len(datTmp[0][0])):
            colPack.extend(np.column_stack((datTmp[i][0][j], datTmp[i][1][j])))  # 2-feature
        
    datStmp = np.array(colPack).reshape(pakNum, 32, 2)
    labStmp = np.array(dat2Stmp).reshape(pakNum, 32, 1)
    return datStmp, labStmp

if __name__ == "main":
    main()