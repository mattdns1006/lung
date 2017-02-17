import numpy as np
import pdb
np.set_printoptions(precision=3,linewidth=300)

class Cubify():
    def __init__(self,oldshape,newshape):
        self.newshape = np.array(newshape)
        self.oldshape = np.array(oldshape)
        self.repeats = (oldshape / newshape).astype(int)
        self.tmpshape = np.column_stack([self.repeats, newshape]).ravel()
        order = np.arange(len(self.tmpshape))
        self.order = np.concatenate([order[::2], order[1::2]])
        self.reverseOrder = self.order.copy()
        self.reverseOrder[1:-1] = self.reverseOrder[1:-1][::-1]
        self.reverseReshape = np.concatenate([self.repeats,self.newshape])

    def cubify(self,arr):
        # newshape must divide oldshape evenly or else ValueError will be raised
        arr = arr.reshape(self.tmpshape)
        arr = arr.transpose(self.order)
        arr = arr.reshape(-1, *self.newshape)
        return arr

    def uncubify(self,arr):
        return arr.reshape(self.reverseReshape).transpose(self.reverseOrder).reshape(self.oldshape)

if __name__ == "__main__":
    import itertools
    N = 9
    x = np.arange(N**3).reshape(N,N,N)
    oldshape = x.shape
    newshape = np.array([3,3,3])
    cuber = Cubify(oldshape,newshape)
    out = cuber.cubify(x)
    back = cuber.uncubify(out)
    transpose = [0,2,4,1,3,5]
    #possibleTransposes = set(itertools.permutations(transpose))
    #for perm in possibleTransposes:
    #    reshape = out.reshape(2,2,2,2,2,2).transpose(perm).flatten() 
    #    increasing = np.all(np.diff(reshape)>=0)
    #    if increasing == True:
    #        print(increasing)
    #        print(perm)
    ##back = out.reshape(2,2,2,2,2,2).transpose(transpose).reshape(N,N,N)

    pdb.set_trace()
        

