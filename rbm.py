import numpy as np
import unittest

def relu(x): return max(0.0, x)

class RBM:
  def __init__(self, weights, shape=(1,1), activation_function=relu, bias=0):
    """Shape[0] - input vector length
    Shape[1] - hidden layer size
    """
    self.shape = shape
    if weights.any():
      assert shape == weights.shape
      self.weights = weights
    else:
      self.weights = np.random.rand(shape[0], shape[1])
    self.activation_function = activation_function
    self.bias = bias

  def forward(self, input):
    assert self.shape[0] == len(input)
    result = []
    for c in range(self.shape[1]):
      result.append(self.activation_function(sum(self.weights[:,c] * np.array(input)) + self.bias))
    return result

class TestRBM(unittest.TestCase):
  def test_simple(self):
    r = RBM(shape=(2,2), bias=1, weights=np.array([1,2,3,4]).reshape((2,2)))
    res = r.forward([1,2])
    self.assertEqual(res[0], 8)
    self.assertEqual(res[1], 11)

if __name__ == '__main__':
  unittest.main()
  