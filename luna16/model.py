import tensorflow as tf
import numpy as np
import sys
sys.path.append("/home/msmith/misc/tfFunctions")
from batchNorm2 import bn

def imSum(name,img,max_outputs=10):
    tf.summary.image(name,img,max_outputs)

def W(shape,weightInit,scale=0.05):
    if len(shape) == 5:
        fw, fh, fd, nIn, nOut = shape
        fan_in = fw*fh*fd*nIn # Number of weights for one output neuron in nOut
        fan_out = nOut #  
    else:
        raise ValueError, "Not a valid 3D shape"

    if weightInit == "uniform":
        init = tf.random_uniform(shape,minval=-scale,maxval=scale)

    elif weightInit == "normal":
        init = tf.random_normal(shape,mean=0,stddev=scale)

    elif weightInit == "lecun_uniform":
        scale = np.sqrt(3.0/(fan_in))
        init = tf.random_uniform(shape,minval=-scale,maxval=scale)

    elif weightInit == "glorot_normal":
        scale = np.sqrt(2.0/(fan_in+fan_out))
        init = tf.random_normal(shape,mean=0,stddev=scale)

    elif weightInit == "glorot_uniform":
        scale = np.sqrt(6.0/(fan_in+fan_out))
        init = tf.random_uniform(shape,minval=-scale,maxval=scale)

    elif weightInit == "zeros":
        init = tf.zeros(shape)
    else:
        raise ValueError, "{0} not a valid weight intializer.".format(weightInit)

    return tf.Variable(init)

def B(shape):
    return tf.Variable(tf.constant(0.0,shape=[shape]))

def convolution3d(inTensor,inFeats,outFeats,filterSize,stride=1):
    with tf.name_scope("conv3d"):
        with tf.name_scope("w"):
            weight = W([filterSize,filterSize,filterSize,inFeats,outFeats],"lecun_uniform")
        with tf.name_scope("b"):
            bias = B(outFeats)
        with tf.name_scope("conv"):
            out = tf.nn.conv3d(inTensor,weight,strides=[1,stride,stride,stride,1],padding='SAME') + bias
    return out

def convolution3d_transpose(inTensor,output_shape,inFeats,outFeats,filterSize,stride=1):
    with tf.name_scope("conv3d_transpose"):
        with tf.name_scope("w"):
            weight = W([filterSize,filterSize,filterSize,outFeats,inFeats],"lecun_uniform")
        with tf.name_scope("b"):
            bias = B(outFeats)
        with tf.name_scope("conv"):
            out = tf.nn.conv3d_transpose(inTensor,weight,output_shape,strides=[1,stride,stride,stride,1],padding='SAME') + bias
    return out

def model0(x,is_training,initFeats=16,featsInc=0,nDown=6,filterSize=3,decay=0.95,dropout=1.0):
    imSum("x",x)
    bS = get_shape(x)[0]
    af = tf.nn.relu
    print(x.get_shape())
    dilation = 2
    with tf.variable_scope("convIn"):
        x1 = af(bn(convolution3d(x,1,initFeats,3,stride=2),is_training=is_training,name="bn_0",decay=decay))

    for block in range(nDown):
        if block == 0:
            inFeats = initFeats 
            outFeats = initFeats + featsInc
        else:
            inFeats = outFeats 
            outFeats = outFeats + featsInc
        with tf.variable_scope("block_down_{0}".format(block)):
	    x2 = af(bn(convolution3d(x1,inFeats,outFeats,1,stride=1),is_training=is_training,name="bn_{0}_0".format(nDown),decay=decay))
	    x3 = af(bn(convolution3d(x1,inFeats,outFeats,3,stride=1),is_training=is_training,name="bn_{0}_1".format(nDown),decay=decay))
            x4 = bn(x2+x3,is_training=is_training,name="bn_{0}_3".format(nDown))
	    x1 = tf.nn.max_pool3d(x4,[1,3,3,3,1],[1,2,2,2,1],"SAME")
    	    print(x1.get_shape())

    #x1 = convolution3d_transpose(x1,[1,10,10,10,1],outFeats,1,3)
    #weight = W([filterSize,filterSize,filterSize,1,outFeats],"lecun_uniform")
    #output_shape = [1,10,10,10,1]
    #stride = 2
    #x1 = tf.nn.conv3d_transpose(x1,weight,output_shape,strides=[1,stride,stride,stride,1],padding='SAME') 
    #print(x1.get_shape())

    shape = get_shape(x1)
    bS, depth, height, width, c = shape
    assert depth==height==width, "Must be doing a cube i.e. h = w = d"
    upSample = 2
    depth *= upSample
    height *= upSample
    width *= upSample
    inFeats = outFeats
    pdb.set_trace()
    x1 = convolution3d_transpose(x1,output_shape=[bS,depth,height,width,1],inFeats = outFeats , outFeats = 1,filterSize= 3,stride = upSample)

    yPred = x1
    return yPred

def get_shape(tensor):
    return tensor.get_shape().as_list()

if __name__ == "__main__":
    import pdb
    import numpy as np
    bS = 5
    X = tf.placeholder(tf.float32,shape=[bS,70,70,70,1])
    is_training = tf.placeholder(tf.bool,name="is_training")
    Y = model0(X,is_training=is_training,initFeats=16,featsInc=16,nDown=3)
    weights = [w for w in tf.trainable_variables() if "/w/" in w.name]
    with tf.Session() as sess:
        tf.global_variables_initializer().run()
        for i in xrange(10):
            x = np.random.rand(5,70,70,70,1)
            y_ = sess.run([Y],feed_dict={X:x,is_training.name:True})[0]
            print(y_.shape)
            if i == 9:
                pdb.set_trace()


