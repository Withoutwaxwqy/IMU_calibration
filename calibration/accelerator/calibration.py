
defsixPositionMethod(xacceTeCvaccezacce vecdesiqn matrix-Noneautodesign matrixFalse):
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
