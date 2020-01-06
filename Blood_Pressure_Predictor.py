## GUI building :

# Import self-lib :
from keras.layers import Input
import BPModel as Bp
import Data_Preprocessing as DPre
import realPlot

# Import 3-part lib for GUI and plotting:
from tkinter import filedialog
import tkinter as tk
import numpy as np
# For plotting the result
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
#%matplotlib qt

## self-definition function :
def load_data():
    file_path =  filedialog.askopenfilename(initialdir = "./",title = "Select file",filetypes = (("matlab files","*.mat"), ("all files","*.*")))
    traSetLst, tstSetLst = DPre.preprocess(DPre.load_data(file_path, datVar.get()), seq_len, sample_num)  ## dbNam, datNam
    global traDatStmp, traLabStmp
    global tstDatStmp, tstLabStmp
    traDatStmp, traLabStmp = DPre.pack_time_stamp(traSetLst, fea_num, len(traSetLst))         ## package train data..
    tstDatStmp, tstLabStmp = DPre.pack_time_stamp(tstSetLst, fea_num, len(tstSetLst))         ## package test data..
    predBut.config(state="normal")
    global model
    input = Input(shape=(seq_len, fea_num))  # sample num : 2000, time_stamp : 32(seq_len), feature num : 2 
    model = Bp.build_model(input)
    try:
        model.load_weights('./model_save/model_weight/'+datVar.get()+'.h5')
        msg.set('load success!')
        label.config(text=msg.get())  
    except:
        msg.set('fail to load')
        label.config(text=msg.get())  
    
def init():
    global X, Y, Z
    X, Y, Z = [], [], []
    global xlim, ln1, ln2
    xlim = []
    xlim.append(0)
    xlim.append(150)
    ax.set_xlim(xlim[0], xlim[1])
    ax.set_ylim(0, 180)
    ax.set_title("Blood Pressure Prediction  animation (real time)", fontdict={'fontsize':15})
    ax.set_xlabel('time axis', fontdict={'fontsize':12})
    ax.set_ylabel('ABP value', fontdict={'fontsize':12})
    ln1, ln2, = ax.plot([], [], '-r', [], [], '-b', animated=False)
    ax.legend((ln1, ln2), ('Predict BP', 'Ground Truth'))
    return ln1, ln2,

def update(frame):
    Y.append(frame[0])              
    Z.append(frame[1])
    X.append(len(Y))
    if len(X) > 150:            
        xlim[0]+=1
        xlim[1]+=1
        ax.set_xlim(xlim[0],xlim[1])
    ln1.set_data(X, Y)
    ln2.set_data(X, Z)
    return ln1, ln2,
        
def predict():
    test_output = model.predict(tstDatStmp, verbose=2)
    global fig, ax
    fig, ax = plt.subplots()
    ani = FuncAnimation(fig, func=update, init_func=init, interval=20, frames=np.column_stack((test_output.reshape(-1), tstLabStmp.reshape(-1))), blit=True, repeat=False)
    plt.show()
    
## Declare the parameters of input/output data
fea_num = 2
seq_len = 32   # sequence len (time_stamp), so seq_len -> pred 1 output value
sample_num = 2000 
datNum = seq_len*sample_num # total num of sample..

## main window definition :    
win = tk.Tk()
win.title("Blood Pressure Prediction")
win.geometry("800x600")

## Variable Declare : 
datVar = tk.StringVar()
msg = tk.StringVar()
msg.set("module unload..")
## Button Declare and Pack into :
tk.Label(win, text='The Blood Pressure Predictor : ', font=("Helvetica", 30), fg="blue").pack()

tk.Label(win, text='Please type the data variable name :', font=("Helvetica", 15), fg="red").pack()
tk.Entry(win, textvariable=datVar).pack()
tk.Button(win, text='Load Blood \n Pressure data', width='15', height='5', command=load_data).pack()

label = tk.Label(win, text=msg.get(), font=("Helvetica", 15), fg="red")
label.pack()

predBut = tk.Button(win, text='Real time \n predict', width='15', height='5', command=predict)
predBut.pack()
predBut.config(state="disabled")

#---------------------------------------------------------------------------------------
# testing phase :  

win.mainloop()

if __name__ == "main":
    main()