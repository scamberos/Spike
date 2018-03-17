import os
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'

# python 01_NN.py
# execfile( '01_NN.py' )

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import scipy.io
from scipy import misc
import tensorflow as tf
from sklearn import preprocessing as preprocess


# Music type classifier


# NN basic building block

def FullyConnected( layerInput, layerName, filterShape, reluFlag=True ):
    with tf.variable_scope( layerName ):
        W_fc = tf.get_variable( name='weights', initializer=tf.truncated_normal( filterShape, stddev=0.1 ) )
        b_fc = tf.get_variable( name='biases', initializer=tf.constant( 0.1, shape=[filterShape[1]] ) )

        fcl = tf.matmul( layerInput, W_fc ) + b_fc
        
        if reluFlag == True:
            layerOutput = tf.nn.relu( fcl )
        else:
            layerOutput = fcl
        
    return layerOutput


# reset default graph
tf.reset_default_graph()


if 1:
    # read train input dataset
    df = pd.read_csv( "data_train.csv" )
    print 'Train input dataset dtypes'
    print df.dtypes
    print

    musicTypeDict = { "music type": {"reggaeton":1, "otros": 0} }
    df.replace( musicTypeDict, inplace=True )

    newTypes = { 'music type': np.float }
    df = df.astype( newTypes )

    # full variable set
    #subset = [ 'acousticness', 'danceability', 'duration', 'energy', 'id_new', 'instrumentalness', 'key', 'liveness', 'loudness', 'mode', 'music type', 'popularity', 'speechiness', 'tempo', 'time_signature', 'valence' ]

    # features that make a reggeaton type of music
    
    # model 1
    inputSubset = [ 'acousticness', 'danceability', 'energy', 'instrumentalness', 'key', 'speechiness', 'tempo', 'valence' ]
    # model 2
    #inputSubset = [ 'danceability', 'energy', 'instrumentalness', 'key', 'speechiness', 'tempo', 'valence' ]
    # model 3
    #inputSubset = [ 'danceability', 'energy', 'instrumentalness', 'key', 'speechiness', 'valence' ]
    # model 4
    #inputSubset = [ 'danceability', 'energy', 'instrumentalness', 'key', 'valence' ]

    # mean subtraction and normalization
    if 1:
        for field in inputSubset:
            df[field] = (df[field] - df[field].mean()) / df[field].std()
            #print df[field].mean()
            #print df[field].std()
            #print

    data_train_input = df[inputSubset].values


    outputSubset = ['music type']

    hotEncoder = preprocess.OneHotEncoder( n_values=2 )
    encoderReference = np.array( [[1], [0]] )
    hotEncoder.fit( encoderReference )
    encoderData = np.reshape( df[outputSubset].values, (-1, 1) )
    # check code: 01 reggaeton, 10 otros
    print 'Hot encode for 1: reggaeton', hotEncoder.transform( [[1]] ).toarray()
    print 'Hot encode for 0: otros', hotEncoder.transform( [[0]] ).toarray()
    print

    data_train_output = hotEncoder.transform( encoderData ).toarray()
    
    samplesTraining = data_train_input.shape[0]

    print( "Train input shape", data_train_input.shape )
    print( "Train output shape", data_train_output.shape )
    print


if 1:
    # read test dataset
    df_t = pd.read_csv( "data_test_proc.csv" )
    
    # mean subtraction and normalization
    if 1:
        for field in inputSubset:
            df_t[field] = (df_t[field] - df_t[field].mean()) / df_t[field].std()
            #print df_t[field].mean()
            #print df_t[field].std()
            #print
    
    data_test_input = df_t[inputSubset].values
    
    samplesTesting = data_test_input.shape[0]

    print( "Test input", data_test_input.shape )
    print


# NN graph build
print 'Neural network graph build'
print

# input and output placeholders for training
with tf.name_scope('input'):
    # multidimensional vector
    inputVariables = len( inputSubset )
    input_Actual = tf.placeholder( dtype=tf.float32, shape=(None, inputVariables), name="input_actual" )
    layer0 = tf.identity( input_Actual )

with tf.name_scope('output'):
    # classifier expected output
    outputClass = data_train_output.shape[1]
    output_Actual = tf.placeholder( dtype=tf.float32, shape=(None, outputClass), name="output_actual" )
    outputActual = tf.identity( output_Actual )


layers = 4
#layers = 8

density = 1
#density = 2


weightsDict = { "wl1": [inputVariables, 1024*density],
    "wl2": density * np.array( [1024, 512] ),
    "wl3": density * np.array( [512, 256] ),
    "we4": [256*density, 2],
    "wl4": density * np.array( [256, 128] ),
    "wl5": density * np.array( [128, 64] ),
    "wl6": density * np.array( [64, 32] ),
    "wl7": density * np.array( [32, 16] ),
    "we8": [16*density, 2] }

with tf.variable_scope('model'):
    # fully connected layers
    with tf.variable_scope('fc'):
        # Layers
        if layers == 4:
            layer1 = FullyConnected( layer0, 'L1', weightsDict["wl1"] )
            layer2 = FullyConnected( layer1, 'L2', weightsDict["wl2"] )
            layer3 = FullyConnected( layer2, 'L3', weightsDict["wl3"] )
            layer4 = FullyConnected( layer3, 'L4', weightsDict["we4"], reluFlag=False )
            layerLogits = tf.identity( layer4, name="layer_logits" )

        if layers == 8:
            layer1 = FullyConnected( layer0, 'L1', weightsDict["wl1"] )
            layer2 = FullyConnected( layer1, 'L2', weightsDict["wl2"] )
            layer3 = FullyConnected( layer2, 'L3', weightsDict["wl3"] )
            layer4 = FullyConnected( layer3, 'L4', weightsDict["wl4"] )
            layer5 = FullyConnected( layer4, 'L5', weightsDict["wl5"] )
            layer6 = FullyConnected( layer5, 'L6', weightsDict["wl6"] )
            layer7 = FullyConnected( layer6, 'L7', weightsDict["wl7"] )
            layer8 = FullyConnected( layer7, 'L8', weightsDict["we8"], reluFlag=False )
            layerLogits = tf.identity( layer8, name="layer_logits" )

    # softmax layer
    with tf.variable_scope('softmax'):
        # 2 input, 2 output
        outputPred = tf.nn.softmax( layerLogits, name="output_pred" )


    #logits = 0
    logits = 1

    # cost function
    with tf.variable_scope('cost'):
        if logits == 0:
            # log(x) for x=0 should throw an error
            # either x should be very small or use a different label
            # Tensorflow documentation states this cost formulation is numerically unstable
            # normalization avoids NaN output on this operation
            loss_op = tf.reduce_mean( -tf.reduce_sum( outputActual*tf.log(outputPred), reduction_indices=[1] ), name="cross_entropy_op")
        
        if 0:
            loss_op = tf.reduce_mean( -tf.reduce_sum( outputActual*tf.log(outputPred + 1e-10), reduction_indices=[1] ), name="cross_entropy_op" )

        if logits == 1:
            loss_op = tf.reduce_mean( tf.nn.softmax_cross_entropy_with_logits(labels=outputActual, logits=layerLogits), name="cross_entropy_op" )

    # optimizer
    optimizerEngine = 'Adam'
    #optimizerEngine = 'Gradient'

    with tf.variable_scope('train'):
        if optimizerEngine == 'Adam':
            # Adam optimizer
            # default values: 0.001 learningRate, 0.9 beta1, 0.999 beta2, 1e-08 epsilon
            #learningRate = 1e-6
            learningRate = 1e-3
            beta1 = 0.9
            beta2 = 0.999
            #epsilon = 1e-08
            epsilon = 1e-06
            #epsilon = 1e-03
            optimizer = tf.train.AdamOptimizer( learningRate, beta1, beta2, epsilon  )
            training_op = optimizer.minimize( loss_op )

        if optimizerEngine == 'Gradient':
            # gradient descent optimizer
            #learningRate = 1e-06
            learningRate = 1e-03
            #learningRate = 1e-01
            optimizer = tf.train.GradientDescentOptimizer( learningRate )
            training_op = optimizer.minimize( loss_op )


    # accuracy
    with tf.variable_scope('accuracy'):
        # tf.argmax, returns the index with the largest value across axes of a tensor
        # counts matching indexes w/highest value (either 1 or probability), the bigger the better
        correct_prediction = tf.equal( tf.argmax(outputPred, axis=1), tf.argmax(outputActual, axis=1) )
        accuracy_op = tf.reduce_mean( tf.cast( correct_prediction, tf.float32), name="accuracy_op" )


# initialize variables and create Tensorflow session
print( 'Start session' )
print
init = tf.global_variables_initializer()
sess = tf.Session()
sess.run( init )


# training track
opSummary_cost = tf.summary.scalar( "sclr_cost", loss_op )
opSummary_accuracy = tf.summary.scalar( "sclr_accuracy", accuracy_op )

# write for graph visualization
merged = tf.summary.merge_all()
writer = tf.summary.FileWriter( "summary_logs", sess.graph )


print 'Start training'
print

# training epochs quantity
#epochs = 512
epochs = 384
#epochs = 256
#epochs = 128
iterations = 12
batchSize = 191

loss_previous = 0.0
loss_change = 0.0


# epoch iterations
for k in range( epochs ):
    # training iterations
    for i in range( iterations ):
        j = i*batchSize
        train_input = data_train_input[j:j+batchSize, :]
        train_output = data_train_output[j:j+batchSize, :]
        
        # train step
        train_feed = { input_Actual: train_input, output_Actual: train_output }

        summary, result, loss, accuracy = sess.run( [merged, training_op, loss_op, accuracy_op], feed_dict=train_feed )

        if i>0:
            loss_new = loss
            loss_change = abs( loss_new - loss_previous ) / loss_previous

        if i==0:
            writer.add_summary( summary, k )
        
            if (k+1)%10 == 0:
                print 'epoch', k+1
                print 'iteration', i+1
                print 'loss', loss
                print 'loss change', loss_change
                print 'accuracy', accuracy
                print

        loss_previous = loss


# accuracy over testing set
if 1:
    # group sample accuracy is a better performance measure than individual sample accuracy
    index = 0
    batchSize = samplesTesting
    test_input = data_test_input[index:index+batchSize, :]
    
    test_feed = { input_Actual: test_input }
    
    layerOutput, logitsOutput = sess.run( [outputPred, layerLogits], feed_dict=test_feed )

    print 'Inference testing set shape', layerOutput.shape
    print 'Inference testing set contents', layerOutput
    print

    cutoff = 0.8
    df_output = pd.DataFrame( layerOutput )
    df_output.columns = ['prob_otros', 'prob_reggaeton']
    df_output['marca_reggaeton'] = 0
    #df_output['marca_reggaeton'][df_output['prob_reggaeton'] > cutoff] = 1
    df_output.loc[df_output['prob_reggaeton'] > cutoff, 'marca_reggaeton'] = 1
    print df_output.shape
    print df_output.head()
    print
    
    labels = df_output['marca_reggaeton'].values

    fig, auxArray = plt.subplots( nrows=1, ncols=3, sharex='none', sharey='none', figsize=(10, 5) )
    #auxArray[0].scatter( x=logitsOutput[:, 0], y=logitsOutput[:, 1], c='gray', alpha=0.6, edgecolors='none' )
    auxArray[0].scatter( x=logitsOutput[:, 0], y=logitsOutput[:, 1], c=labels, alpha=0.6, edgecolors='none' )
    #auxArray[1].scatter( x=layerOutput[:, 0], y=layerOutput[:, 1], c='gray', alpha=0.6, edgecolors='none' )
    auxArray[1].scatter( x=layerOutput[:, 0], y=layerOutput[:, 1], c=labels, alpha=0.6, edgecolors='none' )
    auxArray[1].set_xlim( xmin=-0.1, xmax=1.1 )
    auxArray[1].set_ylim( ymin=-0.1, ymax=1.1 )
    auxArray[2].hist( layerOutput[:, 1], color='orange' )
    auxArray[2].set_xlim( xmin=-0.1, xmax=1.1 )
    plt.show()
    
    print 'Records overall', df_output.shape[0]
    print 'Records prob_reggaeton', df_output['marca_reggaeton'].sum()
    print 'Records prob_otros', df_output.shape[0] - df_output['marca_reggaeton'].sum()
    print

    df_t = pd.concat( [df_t, df_output], axis=1 )
    print df_t.shape
    #print df_t.head()
    print
    
    # save inference
    df_t.to_csv( 'data_test_inference.csv' )


# accuracy over training sets
if 1:
    # data separation and accuracy over training set
    index = 0
    batchSize = samplesTraining
    train_input = data_train_input[index:index+batchSize, :]
    train_output = data_train_output[index:index+batchSize, :]
    
    test_feed = { input_Actual: train_input, output_Actual: train_output }
    
    decoded_op = tf.argmax( train_output, axis=1 )

    layerOutput, logitsOutput, labels, accuracy = sess.run( [outputPred, layerLogits, decoded_op, accuracy_op], feed_dict=test_feed )

    print 'Inference training set shape (all)', layerOutput.shape
    #print 'Inference training set contents', layerOutput
    print 'Accuracy over training set (all)', accuracy
    print
    
    fig, auxArray = plt.subplots( nrows=1, ncols=3, sharex='none', sharey='none', figsize=(10, 5) )
    
    df_output = pd.DataFrame( np.concatenate([logitsOutput, np.reshape(labels, (-1, 1))], axis=1) )
    df_output.columns = ['xOutput', 'yOutput', 'labels']
    
    xScatter = df_output['xOutput'][df_output['labels'] == 1]
    yScatter = df_output['yOutput'][df_output['labels'] == 1]
    auxArray[0].scatter( x=xScatter, y=yScatter, c='r', label='reggaeton', alpha=0.6, edgecolors='none' )
    
    xScatter = df_output['xOutput'][df_output['labels'] == 0]
    yScatter = df_output['yOutput'][df_output['labels'] == 0]
    auxArray[0].scatter( x=xScatter, y=yScatter, c='b', label='otros', alpha=0.6, edgecolors='none' )

    auxArray[0].legend( loc='lower left', scatterpoints = 1, fontsize='small', borderpad=0.4, labelspacing=0.4 )
    
    
    df_output = pd.DataFrame( np.concatenate([layerOutput, np.reshape(labels, (-1, 1))], axis=1) )
    df_output.columns = ['xOutput', 'yOutput', 'labels']
    
    xScatter = df_output['xOutput'][df_output['labels'] == 1]
    yScatter = df_output['yOutput'][df_output['labels'] == 1]
    auxArray[1].scatter( x=xScatter, y=yScatter, c='r', label='reggaeton', alpha=0.6, edgecolors='none' )
    
    xScatter = df_output['xOutput'][df_output['labels'] == 0]
    yScatter = df_output['yOutput'][df_output['labels'] == 0]
    auxArray[1].scatter( x=xScatter, y=yScatter, c='b', label='otros', alpha=0.6, edgecolors='none' )
    
    auxArray[1].legend( loc='lower left', scatterpoints = 1, fontsize='small', borderpad=0.4, labelspacing=0.4 )
    
    auxArray[1].set_xlim( xmin=-0.1, xmax=1.1 )
    auxArray[1].set_ylim( ymin=-0.1, ymax=1.1 )
    

    # accuracy over individual training set (reggaeton)
    data_train_input = df[df['music type'] == 1.0][inputSubset].values

    encoderData = np.reshape( df[df['music type'] == 1.0][outputSubset].values, (-1, 1) )
    data_train_output = hotEncoder.transform( encoderData ).toarray()

    index = 0
    batchSize = data_train_input.shape[0]
    train_input = data_train_input[index:index+batchSize, :]
    train_output = data_train_output[index:index+batchSize, :]

    test_feed = { input_Actual: train_input, output_Actual: train_output }

    layerOutput, accuracy = sess.run( [outputPred, accuracy_op], feed_dict=test_feed )

    print 'Inference training set shape (reggaeton)', layerOutput.shape
    #print 'Inference training set contents (reggaeton)', layerOutput
    print 'Accuracy over training set (reggaeton)', accuracy
    print

    auxArray[2].hist( layerOutput[:, 1], color='orange' )
    auxArray[2].set_xlim( xmin=-0.1, xmax=1.1 )
    plt.show()


# writer close
writer.close()

# session close
sess.close()



