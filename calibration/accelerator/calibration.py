import pandas as pd
import matplotlib.pyplot as plt
import re
import matplotlib.ticker as mticker
import matplotlib
import numpy as np

class INSCalibration:
  def __init__(self, herz):
    self.index = []
    self.herz = herz
    self.data7 = pd.DataFrame()

  def plot7columndata(self, path):
    lw = 0.3
    gyrocolor, accecolor = 'red', 'blue'
    fig, axes = plt.subplots(3, 2, figsize=(20.5))



def sixPositionMethod(xacce_vec,yacce_vec,zacce_vec,design_matrix=None,autodesign_matrix=False):
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
    A = desian_matrix
  AAA =A.T @(np.linalg.inv(A *A.T))
