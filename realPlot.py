## realPlot.py
# For plotting the result
import matplotlib.pyplot as plt
from matplotlib import animation

# Plotting the val_acc, acc graph
def training_vis(hist):
    acc = hist.history['rmse']
    val_acc = hist.history['val_rmse']
    
    fig, ax = plt.subplots()
    ax.plot(acc,label='rmse')
    ax.plot(val_acc, label='val_rmse')
    ax.set_xlabel('Epochs')
    ax.set_ylabel('Accuracy')
    ax.set_title('Accuracy  on Training and Validation Data')
    ax.legend()
    plt.tight_layout()

def update(data):
    plt.cla()                   ## Eliminate the previous plot!
    Y.append(data[0])           
    Z.append(data[1])
    X.append(len(Y))
    if len(X) > 150:            ## Update the plot range
        xlim[0]+=1
        xlim[1]+=1
    plt.xlabel('time axis')
    plt.ylabel('ABP value')
    plt.plot(X, Y, label='predict')
    plt.plot(X, Z, label='ground_truth')
    plt.title("Blood Pressure Prediction  animation (real time)")
    plt.ylim(0, 150)
    plt.xlim(xlim[0],xlim[1])
    plt.legend()


if __name__ == "main":
    main()