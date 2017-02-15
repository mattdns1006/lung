import cv2,os,sys, glob, pdb
import numpy as np
import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt
import loadData
from model import model0 as model0
import matplotlib.cm as cm
sys.path.append("/Users/matt/misc/tfFunctions/")
import paramCount
#from dice import dice3D
from paramCount import paramCount
import SimpleITK as sitk
from crop import showCrop
from params import *

def dice3D(Y,YPred):
    smooth = 1.0
    y_true_f = tf.reshape(Y,[-1])
    y_pred_f = tf.reshape(YPred,[-1])
    intersection = tf.reduce_sum(y_true_f*y_pred_f)
    return (2.0 * intersection + smooth)/(tf.reduce_sum(y_true_f) + tf.reduce_sum(y_pred_f) + smooth)

def varSummary(var,name):
    with tf.name_scope('summary'):
        tf.summary.scalar(name, var)
        tf.summary.histogram(name, var)

def lossFn(y,yPred,regularization=0,beta=0.00094):
    with tf.variable_scope("loss"):
        loss = tf.mul(dice3D(y,yPred),-1.0) # make it negative
        regLoss = loss
        if regularization == 1:
            weights = [w for w in tf.trainable_variables() if "/w/" in w.name]
            for w in weights:
                regLoss += beta*tf.nn.l2_loss(w)
        varSummary(loss,"loss")
    with tf.variable_scope("regLoss"):
        varSummary(regLoss,"regLoss")
    #with tf.variable_scope("dice"):
    #    diceScore = tf.reduce_mean(dice3D(yPred,y))
    #    varSummary(diceScore,"dice")
    return loss, regLoss 

def trainer(lossFn, learningRate):
    return tf.train.AdamOptimizer(learningRate).minimize(lossFn)

def nodes(batchSize,inSize,trainOrTest,initFeats,incFeats,nDown,num_epochs,augment):
    if trainOrTest == "train":
        csvPath = "csvs/trainCV.csv"
        print("Training on subset.")
        shuffle = True
    elif trainOrTest == "trainAll":
        csvPath = "csvs/train.csv"
        print("Training on all.")
        shuffle = True
    elif trainOrTest == "test":
        csvPath = "csvs/testCV.csv"
        print("Testing on validation set")
        shuffle = True
        num_epochs = 1
    X,Y,xPath = loadData.read(csvPath=csvPath,
            batchSize=batchSize,
            shuffle=shuffle,
            num_epochs = num_epochs,
            augment = augment
            ) #nodes
    is_training = tf.placeholder(tf.bool)
    drop = tf.placeholder(tf.float32)
    YPred = model0(X,is_training=is_training,nDown=nDown,initFeats=initFeats,featsInc=incFeats,dropout=drop)
    loss, regLoss = lossFn(Y,YPred)
    learningRate = tf.placeholder(tf.float32)
    trainOp = trainer(regLoss,learningRate)
    saver = tf.train.Saver()

    return saver,xPath,X,Y,YPred,loss,is_training,trainOp,learningRate, drop

if __name__ == "__main__":
    import pdb
    nEpochs = 3 
    flags = tf.app.flags
    FLAGS = flags.FLAGS 
    flags.DEFINE_float("lr",0.001,"Initial learning rate.")
    flags.DEFINE_float("lrD",1.00,"Learning rate division rate applied every epoch. (DEFAULT - nothing happens)")
    flags.DEFINE_integer("inSize",64,"Size of input image")
    flags.DEFINE_integer("initFeats",8,"Initial number of features.")
    flags.DEFINE_integer("incFeats",16,"Number of features growing.")
    flags.DEFINE_float("drop",0.945,"Keep prob for dropout.")
    flags.DEFINE_integer("aug",1,"Augment.")
    flags.DEFINE_integer("nDown",5,"Number of blocks going down.")
    flags.DEFINE_integer("bS",10,"Batch size.")
    flags.DEFINE_integer("load",0,"Load saved model.")
    flags.DEFINE_integer("trainAll",0,"Train on all data.")
    flags.DEFINE_integer("fit",0,"Fit training data.")
    flags.DEFINE_integer("show",0,"Show for sanity.")
    flags.DEFINE_integer("nEpochs",10,"Number of epochs to train for.")
    flags.DEFINE_integer("test",0,"Just test.")
    batchSize = FLAGS.bS
    load = FLAGS.load
    if FLAGS.fit == 1 or FLAGS.test == 1:
        load = 1
    specification = "{0}_{1:.6f}_{2}_{3}_{4}_{5}_{6:.3f}_{7}".format(FLAGS.bS,FLAGS.lr,FLAGS.inSize,FLAGS.initFeats,FLAGS.incFeats,FLAGS.nDown,FLAGS.drop,FLAGS.aug)
    print("Specification = {0}".format(specification))
    modelDir = "models/" + specification + "/"
    imgPath = modelDir + "imgs/"
    if not FLAGS.fit:
        if not os.path.exists(modelDir):
            os.mkdir(modelDir)
            os.mkdir(imgPath)
    savePath = modelDir + "model.tf"
    trCount = teCount = 0
    trTe = "train"
    assert FLAGS.test + FLAGS.trainAll + FLAGS.fit in [0,1], "Only one of trainAll, test or fit == 1"
    what = ["train"]
    if FLAGS.test == 1:
        what = ["test"]
        load = 1
        FLAGS.nEpochs = 3 
        FLAGS.aug = 0

    for trTe in what:
        tf.reset_default_graph()
        saver,XPath,X,Y,YPred,loss,is_training,trainOp,learningRate,drop = nodes(
            batchSize=FLAGS.bS,
            trainOrTest=trTe,
            inSize = [FLAGS.inSize,FLAGS.inSize],
            initFeats=FLAGS.initFeats,
            incFeats=FLAGS.incFeats,
            nDown=FLAGS.nDown,
            num_epochs=FLAGS.nEpochs,
            augment = FLAGS.aug 
            )

        gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.90)

        merged = tf.summary.merge_all()
        paramCount()

        with tf.Session(config=tf.ConfigProto(gpu_options=gpu_options)) as sess:
            if load == 1:
                print("Restoring {0}.".format(specification))

                saver.restore(sess,savePath)
            else:
                tf.global_variables_initializer().run()
            tf.local_variables_initializer().run()
            trWriter = tf.summary.FileWriter("summary/{0}/train/".format(specification),sess.graph)
            teWriter = tf.summary.FileWriter("summary/{0}/test/".format(specification),sess.graph)
            coord = tf.train.Coordinator()
            threads = tf.train.start_queue_runners(sess=sess,coord=coord)
            count = 0
            try:
                while True:
                    if trTe in ["train","trainAll"]:
                        _, summary = sess.run([trainOp,merged],feed_dict={is_training:True, drop:FLAGS.drop, learningRate:FLAGS.lr})

                        if count % 400 == 0:
                            print("Seen {0} examples".format(count))
                            _, summary,x,y,yPred,path = sess.run([trainOp,merged,X,Y,YPred,XPath],feed_dict={is_training:True, drop:FLAGS.drop, learningRate:FLAGS.lr})
                            for i in xrange(4):
                                wpX = imgPath +"model_train_x_{0}.nrrd".format(i)
                                wpY = imgPath +"model_train_y_{0}.nrrd".format(i)
                                wpYPred = imgPath +"model_train_yPred_{0}.nrrd".format(i)
                                sitk.WriteImage(sitk.GetImageFromArray(x[i]),wpX)
                                sitk.WriteImage(sitk.GetImageFromArray(yPred[i]),wpYPred)
                                sitk.WriteImage(sitk.GetImageFromArray(y[i]),wpY)

                        trCount += batchSize
                        count += batchSize
                        trWriter.add_summary(summary,trCount)


                        if count % 2000 == 0:
                            saver.save(sess,savePath)
                            print("Saved.")

                        if count > 120000:
                            print("Finished training cba")
                            break

                    elif trTe == "test":
                        summary, _ = sess.run([merged,XPath],feed_dict={is_training:False, drop:1.00})
                        if teCount % 400 == 0:
                            print("Seen {0} examples".format(count))
                            summary,x,y,yPred,xPath = sess.run([merged,X,Y,YPred,XPath],feed_dict={is_training:False,drop:1.00})
                            for i in xrange(x.shape[0]):
                                wpX = imgPath +"model_test_x_{0}.nrrd".format(i)
                                wpY = imgPath +"model_test_y_{0}.nrrd".format(i)
                                wpYPred = imgPath +"model_test_yPred_{0}.nrrd".format(i)
                                sitk.WriteImage(sitk.GetImageFromArray(x[i]),wpX)
                                sitk.WriteImage(sitk.GetImageFromArray(yPred[i]),wpYPred)
                                sitk.WriteImage(sitk.GetImageFromArray(y[i]),wpY)
                        teCount += batchSize
                        teWriter.add_summary(summary,teCount)
                        if teCount % 100 == 0:
                            print("Seen {0} test examples".format(teCount))
                            #showBatch(x,y,yPred,wp="{0}/test.jpg".format(imgPath))

                    elif trTe == "fit":
                        x, yPred,fp = sess.run([X,YPred,XPath],feed_dict={is_training:False,drop:1.00})
                        count += x.shape[0]
                        for i in xrange(4):
                            row = fp[i].tolist() + yPred[i].tolist()
                            df.append(row) 
                        if count % 600 == 0:
                            print("Seen {0} examples".format(count))

                    else:
                        break

                    if coord.should_stop():
                        break
            except Exception as e:
                coord.request_stop(e)
            finally:
                coord.request_stop()
                coord.join(threads)
            print("Finished! Seen {0} examples".format(count))

            if trTe == "train":
                lrC = FLAGS.lr
                FLAGS.lr /= FLAGS.lrD
                print("Dropped learning rate from {0} to {1}".format(lrC,FLAGS.lr))
                print("Saving in {0}".format(savePath))
                saver.save(sess,savePath)
            elif trTe == "fit":
                Df = pd.DataFrame(df)
                Df["img"] = Df.img.apply(lambda x: x.split("/")[-1])
                Df.to_csv("submissions/submission_{0}.csv".format(specification),index=0)
                print("Written submission file.")

            sess.close()

