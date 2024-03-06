import pandas as pd
import matplotlib.pyplot as plt
import re
import matplotlib.ticker as mticker
import matplotlib
import numpy as np

G = 9.7936174
class INSCalibration:
  def __init__(self, herz):
    self.index = []
    self.herz = herz
    self.data7 = pd.DataFrame()

  def plot7columndata(self, path):
    lw = 0.3
    gyrocolor, accecolor = 'red', 'blue'
    fig, axes = plt.subplots(3, 2, figsize=(20.5))
    title = [[self.index[1], self.index[4]],
             [self.index[2], self.index[5]],
             [self.index[3], self.index[6]]]
    latextitle = [['$x_{gyro}$', '$x_{acce}$'],
                  ['$y_{gyro}$', '$y_{acce}$'],
                  ['$z_{gyro}$', '$z_{acce}$']]
    colorlist = [[gyrocolor, accecolor],
                 [gyrocolor, accecolor],
                 [gyrocolor, accecolor]]
    for i in range(3):
      for j in range(2):
        axes[i,j].plot(self.data7[self.index[0]], self.data7[self.indexs[i+j*3+1]], linewidth=lw, c=colorlist[i][j])
        axes[i,j].set_title(latextitle[i][j])
        tick_spacing = self.data7.index.size / 5
        axes[i, j].xaxis.set_major_locator(mticker.MultipleLocator(tick_spacing))
    plt.show()



def sixPositionMethod(xacce_vec,yacce_vec,zacce_vec,design_matrix=None,auto_design_matrix=False):
  """
  six position method to acceleration calibration
  :param xacce vec:l*n matrix X-axis acceleratoroutput
  :param yacce vec: l*n matrix Y-axis accelerator output
  :paramzacce vec:l*n matrix Z-axis accelerator output
  :param design matrix: default-->None, if auto design matrix is False, it should be a 4*6 design ma
  :param auto design matrix:
  :return:
  """
  if auto_design_matrix:
    A = np.array([[G,-G,0,0,0,0],
                 [0,0,G,-G,0,0],
                 [0,0,0,0,G,-G],
                 [1, 1, 1, 1,1,1]])
  else:
    A = design_matrix
  AAA =A.T @(np.linalg.inv(A *A.T))

def isEqualorNegative(vec, value, absolute_tolerance, continuity):
  """
  """
  result = {"index":np.zeros(shape=vec.shape),
            "EqualDuration":[],
            "NegativeDuration":[]}
  vec_e=np.isclose(vec.np.ones(shape=vec.shape)*value,    atol=absolute_tolerance)
  vec_n=np.isclose(vec.np.ones(shape=vec.shape)*(-value), atol=absolute_tolerance)
  result['index'] = vec_e * 1 + vec_n *(-1)
  result['EqualDuration']    = continuousDuration(result['index'],  1, continuity)
  result['NegativeDuration'] = continuousDuration(result['index'], -1, continuity)


def continuousDuration(vec, value, threshold):
  pass