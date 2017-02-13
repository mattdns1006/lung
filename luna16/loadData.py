import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import tensorflow as tf
from params import *
import pdb
from crop import showCrop


def getImg(path):
    fileReader = tf.WholeFileReader()
    k, v = fileReader.read(path)
    return v 

def read(csvPath,batchSize=5,num_epochs=1,shuffle=True,augment=0):
    csv = tf.train.string_input_producer([csvPath],num_epochs=num_epochs,shuffle=shuffle)
    reader = tf.TextLineReader(skip_header_lines=1)
    k, v = reader.read(csv)
    defaults = [tf.constant([],dtype = tf.string), 
                tf.constant([],dtype = tf.string) ]
    xPath, yPath = tf.decode_csv(v,record_defaults = defaults)

    x = tf.read_file(xPath)
    x = tf.decode_raw(x,tf.int16)
    x = tf.reshape(x,IN_SIZE) 
    x = tf.expand_dims(x,3)
    x = tf.cast(x,tf.float32)
    x = (tf.sub(x,MIN)/(tf.sub(MAX,MIN)))

    y = tf.read_file(yPath)
    y = tf.decode_raw(y,tf.uint8)
    y = tf.reshape(y,IN_SIZE)
    y = tf.expand_dims(y,3)
    y = tf.cast(y,tf.float32)

    xPath = tf.reshape(xPath,[1])
    in_size = list(IN_SIZE) 
    in_size += [1]
    #if augment == 1:
    #    x = aug(x,inSize)

    Q = tf.FIFOQueue(64,[tf.float32,tf.float32,tf.string],shapes=[in_size,in_size,[1]])
    enQ = Q.enqueue([x,y,xPath])
    QR = tf.train.QueueRunner(
            Q,
            [enQ]*16,
            Q.close(),
            Q.close(cancel_pending_enqueues=True)
            )
    tf.train.add_queue_runner(QR) 
    dQ = Q.dequeue()
    #X,Y,path = tf.train.batch(dQ,batchSize,16,allow_smaller_final_batch=True)
    X,Y,path = tf.train.batch(dQ,batchSize,16)
    return X, Y, path


if __name__ == "__main__":
    import pdb
    X,Y,path = read("csvs/trainCV.csv")
    with tf.Session() as sess:
        tf.global_variables_initializer().run()
        tf.local_variables_initializer().run()
        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(coord=coord,sess=sess)
        max = 0
        min = 0
        count = 0
        try:
            while True:
                if coord.should_stop():
                    break
                count += 1
                x,y,path_ = sess.run([X,Y,path])
		print(x.shape,y.shape,path_)
		pdb.set_trace()
                #if x.min() < min:
                #    min = x.min()
                #if x.max() > max:
                #    max = x.max()
                #print(count,min,max)
        except Exception,e:
            coord.request_stop(e)
        finally:

            coord.request_stop()
            coord.join(threads)
            ("Finished")
