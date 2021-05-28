#LIBRARIES
import pandas as pd
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.layers import Input, LSTM,Dense
from tensorflow.keras.models import Model
from tensorflow.keras.utils import to_categorical
import tensorflow as tf
import plotly.graph_objects as go
import matplotlib.pyplot as plt
from scipy.spatial.distance import cosine
from itertools import groupby
from operator import itemgetter

threshold = 0.15
consecutive_threshold = 5

# convert an array of values into a timeseries of 3 previous steps matrix
def create_timeseries(dataset, y):
    dataX = []
    dataY = []
    for i in range(2,len(dataset)):
        if i%25 > 1:
            a = np.concatenate((dataset[i - 2], dataset[i - 1],dataset[i]), axis=None)
            dataX.append(a)
            dataY.append(y[i])
    return np.array(dataX), np.array(dataY)

#returns ranges of consecutive numbers to a new list
def ranges(nums):
    consec = []
    for k,g in groupby(enumerate(nums), lambda ix: ix[0] - ix[1]):
        consec.append(list(map(itemgetter(1),g)))
    return consec

def build_model(scope, fname=None):
    with tf.variable_scope(scope):
        # build functional model
        visible = Input(shape=(3, 55))
        hidden1 = LSTM(32, return_sequences=True, name='firstLSTMLayer')(visible)
        hidden2 = LSTM(16, name='secondLSTMLayer', return_sequences=True)(hidden1)
        # left branch decides second agent action
        hiddenLeft = LSTM(10, name='leftBranch')(hidden2)
        agent2 = Dense(5, activation='softmax', name='agent2classifier')(hiddenLeft)
        # right branch decides third agent action
        hiddenRight = LSTM(10, name='rightBranch')(hidden2)
        agent3 = Dense(5, activation='softmax', name='agent3classifier')(hiddenRight)

        model = Model(inputs=visible, outputs=[agent2, agent3])

        model.compile(optimizer='adam',
                      loss={'agent2classifier': 'categorical_crossentropy',
                            'agent3classifier': 'categorical_crossentropy'},
                      metrics={'agent2classifier': ['acc'],
                               'agent3classifier': ['acc']})

        if fname is not None:
            model.load_weights(fname)

    return model

def get_logits(model, obs, action, in_length):
    obs = np.reshape(obs, [1, in_length])
    action = np.reshape(action, [1, 1])
    model_input = np.concatenate([obs, action], axis=1)
    logits = np.asarray(model.predict(model_input.astype('float64'), verbose=0))
    return logits


def test():
    #same results for same model, makes it deterministic
    np.random.seed(1234)
    tf.random.set_seed(1234)


    #reading data
    input = np.load("Transition_adv_0.npy", allow_pickle=True)
    #print(input.shape)
    #print(input[0])
    pre = np.asarray(input[:,0])
    a1 = np.asarray(input[:,1])
    a2 = np.asarray(input[:,2])
    a3 = np.asarray(input[:,3])
    post = np.asarray(input[:,4])
    labels = np.asarray(input[:, 5])

    #flattens the np arrays
    pre = np.concatenate(pre).ravel()
    pre = np.reshape(pre, (pre.shape[0]//54,54))
    post = np.concatenate(post).ravel()
    post = np.reshape(post, (post.shape[0]//54,54))

    prea1 = np.column_stack((pre,a1.T))
    a2a3 = np.column_stack((a2.T,a3.T))
    testX = prea1
    testY = a2a3


    #reshapes trainX to be timeseries data with 3 previous timesteps
    #LSTM requires time series data, so this reshapes for LSTM purposes
    #X has 200000 samples, 3 timestep, 55 features
    timex, testY = create_timeseries(testX,testY)
    testX = np.reshape(timex,(timex.shape[0],3,timex.shape[1]//3))

    testX = testX.astype('float64')
    testY = testY.astype('int32')

    model = build_model() #FIXME: add weights

    #setting the threshold
    pred = np.asarray(model.predict(testX.astype('float64'), verbose=0))
    #pred = np.argmax(pred,axis=2)
    #pred = pred.reshape((pred.shape[1],pred.shape[0]))
    pred = pred.reshape((pred.shape[1],pred.shape[0]*pred.shape[2]))
    #print(pred.shape)
    #testY = to_categorical(testY)
    #print(testY.shape)

    testY2, testY3 = np.hsplit(testY,2)
    #print(testY2.shape)

    #creates a simple dataframe in pandas
    test_score_df = pd.DataFrame(pred)
    test_score_df['Action2'] = testY2
    test_score_df['Action3'] = testY3
    test_score_df.rename(columns={"0": "2agent0", "1": "2agent1", "2":"2agent2", "3":"2agent3","4":"2agent4","5": "3agent0", "6": "3agent1", "7":"3agent2", "8":"3agent3","9":"3agent4"})


    """
    #may need to be (a and b) or (c and d)
    anomalous = []
    for i in range(pred.shape[0]):
        anomalous.append(pred[i,test_score_df['Action3'][i]+5] < threshold and pred[i,test_score_df['Action2'][i]+5] < threshold
                            and pred[i,test_score_df['Action3'][i]] < threshold and pred[i,test_score_df['Action2'][i]] < threshold)
    anomalous = np.asarray(anomalous)

    test_score_df['anomaly'] = anomalous

    #IMPLEMENT TIME WINDOW
    anomalies = test_score_df.loc[test_score_df['anomaly']==True]
    print(anomalies)


    consec_ranges = ranges(anomalies.index)
    consec_anomalies = []
    for i in range(len(consec_ranges)):
        if len(consec_ranges[i]) >= consecutive_threshold:
            consec_anomalies.append([consec_ranges[i][0],consec_ranges[i][len(consec_ranges[i])-1]])

    consec_anomalies = np.asarray(consec_anomalies)
    pred_anomalies = np.asarray(np.zeros((anomalous.shape)))
    for i in range(len(consec_anomalies)):
        pred_anomalies[consec_anomalies[i,0]:consec_anomalies[i,1]+1] = 1
    print("Number of anomalies found: ",str(np.count_nonzero(pred_anomalies == 1)))
    print(consec_anomalies)
    indices = list(test_score_df.index)

    #np.save("pred_anomalies_binary.npy",pred_anomalies)
    """


    '''
    #plot for loss and anomalies in environment
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=indices, y=test_score_df['losses'], name='Test loss'))
    fig.add_trace(go.Scatter(x=indices, y=[threshold2] * len(indices), name='Threshold'))
    fig.add_trace(go.Scatter(x=anomalies2.index, y=anomalies2['losses'], mode='markers', name='Anomaly on 2'))
    fig.update_layout(showlegend=True, title='Test loss vs. Threshold for Agent 2')
    fig.show()
    '''
    '''
    #plot for loss and anomalies in environment
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=indices, y=test_score_df['loss3'], name='Test loss for Agent 3'))
    fig.add_trace(go.Scatter(x=indices, y=[threshold3] * len(indices), name='Threshold for 3'))
    fig.add_trace(go.Scatter(x=anomalies3.index, y=anomalies3['loss3'], mode='markers', name='Anomaly on 3'))
    fig.update_layout(showlegend=True, title='Test loss vs. Threshold for Agent 3')
    fig.show()
    '''

if __name__ == "__main__":
    test()