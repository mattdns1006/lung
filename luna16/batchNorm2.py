from tensorflow.contrib.layers.python.layers import batch_norm as batch_norm
import tensorflow as tf

def bn(x,is_training,name,decay=0.99):
    bn_train = batch_norm(x, decay=decay, center=True, scale=True, updates_collections=None, is_training=True, reuse=None, trainable=True, scope=name)
    bn_infer = batch_norm(x, decay=0.0, center=True, scale=True, updates_collections=None, is_training=False, reuse=True, trainable=False, scope=name)
    z = tf.cond(is_training, lambda: bn_train, lambda: bn_infer)
    return z


if __name__ == "__main__":
    print("Example")

    import os
    import numpy as np
    import scipy.stats as stats
    np.set_printoptions(suppress=True,linewidth=200,precision=3)
    np.random.seed(1006)
    import pdb
    path = "batchNorm/"
    if not os.path.exists(path):
        os.mkdir(path)
    savePath = path + "bn.model"

    nFeats = 2
    X = tf.placeholder(tf.float32,[None,nFeats])
    is_training = tf.placeholder(tf.bool,name="is_training")
    Y = bn(X,is_training=is_training,name="bn")
    mvn = stats.multivariate_normal([0,100])
    bs = 4
    load = 0
    train = 1
    saver = tf.train.Saver()
    def bnCheck(batch,mu,std):
        # Checking calculation
        return (x - mu)/(std + 0.001)
    with tf.Session() as sess:
        if load == 1:
            saver.restore(sess,savePath)
        else:
            tf.global_variables_initializer().run()
        if train == 1:
            for i in xrange(100):
                x = mvn.rvs(bs)
                y = Y.eval(feed_dict={X:x, is_training.name: True})

        def bnParams():
            beta, gamma, mean, var = [v.eval() for v in tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES,scope="bn")]
            return beta, gamma, mean, var

        beta, gamma, mean, var = bnParams()
        for i in xrange(10):
            x = mvn.rvs(1).reshape(1,-1)
            check = bnCheck(x,mean,np.sqrt(var))
            y = Y.eval(feed_dict={X:x, is_training.name: False})
            print("x = {0}, y = {1}, check = {2}".format(x,y,check))
            beta, gamma, mean, var = bnParams()
            print("BN Params: Beta {0} Gamma {1} mean {2} var{3} \n".format(beta,gamma,mean,var))

        saver.save(sess,savePath)
