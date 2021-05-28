#LIBRARIES
import numpy as np
from tensorflow.python.keras.layers import LSTM, Dense, Input
from tensorflow.python.keras.models import Model
import tensorflow as tf
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.losses import kullback_leibler_divergence
from tensorflow.keras.losses import CategoricalCrossentropy
from scipy.spatial.distance import cosine



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

        model.summary()

        if fname is not None:
            model.load_weights(fname)

    return model




def train_model():
    #same results for same model, makes it deterministic
    np.random.seed(1234)
    #tf.random.set_seed(1234)


    #reading data
    input = np.load("Transition_Attack_Policy.npy", allow_pickle=True)
    print(input.shape)
    print(input[0])
    pre = np.asarray(input[:,0])
    a1 = np.asarray(input[:,1])
    a2 = np.asarray(input[:,2])
    a3 = np.asarray(input[:,3])
    post = np.asarray(input[:,4])

    #flattens the np arrays
    pre = np.concatenate(pre).ravel()
    pre = np.reshape(pre, (pre.shape[0]//54,54))
    post = np.concatenate(post).ravel()
    post = np.reshape(post, (post.shape[0]//54,54))

    prea1 = np.column_stack((pre,a1.T))
    a2a3 = np.column_stack((a2.T,a3.T))

    prea1 = prea1.astype('float64')
    mean = np.mean(prea1,axis=0)
    prea1 -= mean
    std = np.std(prea1,axis=0)
    std = np.where(std==0, 1, std)
    prea1 /=std
    print(np.mean(prea1,axis=0))
    print(np.std(prea1,axis=0))

    #reshapes trainX to be timeseries data with 3 previous timesteps
    #LSTM requires time series data, so this reshapes for LSTM purposes
    #X has 200000 samples, 3 timestep, 55 features
    timex, inputY = create_timeseries(prea1,a2a3)
    inputX = np.reshape(timex,(timex.shape[0],3,timex.shape[1]//3))


    trainX = inputX[:180000]
    trainY = inputY[:180000]
    valX = inputX[180000:]
    valY = inputY[180000:]
    #testX = inputX[180000:]
    #testY = inputY[180000:]

    trainX = trainX.astype('float64')
    valX = valX.astype('float64')
    trainY = trainY.astype('float64')
    valY = valY.astype('float64')
    #testX = testX.astype('float64')
    #testY = testY.astype('float64')

    print(trainX.shape)
    print(trainY.shape)
    print(valX.shape)
    print(valY.shape)
    #print(testX.shape)
    #print(testY.shape)
    print(np.unique(trainY,return_counts=True))

    #two categorical arrays, one for each side of the functional network
    #testY1, testY2 = np.hsplit(testY,2)
    #testY1 = to_categorical(testY1)
    #testY2 = to_categorical(testY2)
    valY1, valY2 = np.hsplit(valY,2)
    valY1 = to_categorical(valY1)
    valY2 = to_categorical(valY2)
    trainY1, trainY2 = np.hsplit(trainY,2)
    trainY1 = to_categorical(trainY1)
    trainY2 = to_categorical(trainY2)



    model = build_model("Prediction")

    print(model.summary())


    history = model.fit(trainX,
                        y={'agent2classifier': trainY1,'agent3classifier':trainY2}, epochs=1500, batch_size=5000, verbose=2,
                        validation_data = (valX,
                                           {'agent2classifier': valY1,'agent3classifier':valY2}),shuffle=False)

    #save_model(model, 'actionMultiClassNetwork')
    model.save_weights('actionMultiClassNetwork')


    #model = load_model("actionMultiClassNetwork.keras")


    np.save("action_multiclass_history.npy", history.history, allow_pickle=True)

if __name__ == "__main__":
    train_model()