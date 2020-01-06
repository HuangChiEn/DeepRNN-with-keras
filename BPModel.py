# Bi_directional LSTM with blood pressure prediction :
# BP_Model.py
#-----------------------------------------------------------------------------------------------
# Import 3-part package : 

# For building the model
import tensorflow as tf
from keras import backend as K  ## rename for : kernel
from keras import regularizers  ## For L1-regular
from keras.models import Model  ## Using function API (without Sequential structure)
from keras.layers import (Dense, LSTM, Bidirectional, BatchNormalization)      ## For building bidirectional LSTM
from keras.layers.advanced_activations import LeakyReLU
from keras.optimizers import Adam

## For building residual connection blocks
from keras.layers import (Lambda, Input)  
from keras.layers.merge import add

## The RepeatVector module is worse 
## (It's return the last output of previous layer only to pass to next layer)
from keras.layers import TimeDistributed  ## For building multi-input to multi-output(time_stamp)

from keras.preprocessing.sequence import pad_sequences


# self-def loss function by default
def rmse(y_true, y_pred):
        return K.sqrt(K.mean(K.square(y_pred - y_true), axis=-1)) 

def make_residual_lstm_layers(input, rnn_width, rnn_depth, rnn_dropout):
    """
    The intermediate LSTM layers return sequences, while the last returns a single element.
    The input is also a sequence. In order to match the shape of input and output of the LSTM
    to sum them we can do it only for all layers but the last.
    """
    x = input
    for i in range(rnn_depth):
        return_sequences = i < rnn_depth - 1 ## For the last layer not return sequence..
        x_rnn = LSTM(rnn_width, recurrent_dropout=rnn_dropout, dropout=rnn_dropout, return_sequences=True, activation='relu')(x)
        
        if return_sequences:
            # Intermediate layers return sequences, input is also a sequence.
            if i > 0 or input.shape[-1] == rnn_width:
                x = add([x, x_rnn])
            else:
                # Note that the input size and RNN output has to match, due to the sum operation.
                # If we want different rnn_width, we'd have to perform the sum from layer 2 on.
                x = x_rnn
        else:
            # Last layer does not return sequences, just the last element
            # so we select only the last element of the previous output.
            def slice_last(x):
                return x[..., -1, :]
            x = add([Lambda(slice_last)(x), x_rnn])
    return x

# kernel_initializer='glorot_uniform',for lstm
def build_model(input): # 2-B model
    output1 = Bidirectional(LSTM(128, return_sequences=True, activation='relu'), merge_mode='concat')(input)
    output2 = Bidirectional(LSTM(128, return_sequences=True, activation='relu'), merge_mode='concat')(output1)
    output3 = Bidirectional(LSTM(128, return_sequences=True, activation='relu'), merge_mode='concat')(output2)
    output4 = BatchNormalization(momentum=0.3)(output3)
    output5 = make_residual_lstm_layers(output4, rnn_width=100, rnn_depth=3, rnn_dropout=0.4)
    output6 = LSTM(128, return_sequences=True, activation='relu')(output5)
    output7 = BatchNormalization(momentum=0.3)(output6)
    output8 = TimeDistributed(Dense(1, activity_regularizer=regularizers.l1(0.01)))(output7)
    model = Model(inputs=input, outputs=output8)
    opt = Adam(lr=0.001)
    model.compile(optimizer=opt, loss = 'mse', metrics=[rmse])
    return model


if __name__ == "main":
    main()