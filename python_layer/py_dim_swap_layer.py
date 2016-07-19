import caffe
import numpy as np

class DimensionSwapLayer(caffe.Layer):

  def setup(self, bottom, top):
    assert(len(bottom) == 1)
    assert(len(bottom) == len(top))

    self.order = [int(x) for x in self.param_str.split(',')]

  def reshape(self, bottom, top):
    bshape = bottom[0].data.shape
    tshape = [bshape[i] for i in self.order]
    
    top[0].reshape(*tshape)
    assert(len(self.order) == len(bottom[0].data.shape))
    tmp = range(len(self.order))
    self.reverse_order = [self.order.index(i) for i in tmp]
    


  def forward(self, bottom, top):

    top[0].data[:] = np.transpose(bottom[0].data[:], self.order)
    
  def backward(self, top, prop_down, bottom):

    if (prop_down[0]):
      bottom[0].diff[:] = np.transpose(top[0].diff[:], self.reverse_order)