import pandas as pd
import matplotlib.pyplot as plt
import re
import matplotlib.ticker as mticker
import matplotlib
import numpy as np

G = 9.7936174
class INSCalibrationWithoutT:
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

class INSCalibration:
# 在标定前已知的信息 几个位置 位置设计矩阵 
# 最后需要喂进去的是几个位置平均值，零位置平均值， 还有速度， 暂时只考虑这么多
# 初始的数据矩阵有
    def __init__(self, herz:int, posi_number:int, posi_matrix, temp_list):
        """
        initialize the calibration system-------------------------->
        :param herz: frequency of device
        :param posi_number:
        :param posi_matrix: design_matrix which should be {posi_number x 3} such as [[-1, 0, 0]..[0, 1, 0]]
        :param temp_list:
        """
        self.index = []
        self.herz = herz
        self.posinumber  = posi_number
        self.posi_matrix = posi_matrix
        # for 温度系数改正
        self.temp_list   = temp_list # degree exp: [-20, -10, 0, 10, 20 ... N] if len(temp_list)== 1 It can be considered as normal temp
        self.num_temp_sensor = 2
        # for gyro
        self.Pdata = pd.DataFrame() # ALL posi data (maybe not used) should be {(temp) x (posi) x (time) x (sensors)}
        self.Rdata = pd.DataFrame() # ALL velc data (maybe not used) should be {(temp) x (posi) x (time) x (sensors)}
        self.Sp    = pd.DataFrame() # position average matrix --- should be {(temp) x (posinumber*3+1)} 1 is temp sensor--> 
        self.Sr    = pd.DataFrame() # velocity average matrix --- should be {(temp) x (posinumber*3+1)}
        # for acce

    def check(self):
        pass

    def data2SpSr(self):
        Pdata = np.mean(self.Pdata, axis=2)
        # Pdata.reshape(())
        Rdata = np.mean(self.Rdata, axis=2)
        self.Sp = pd.DataFrame(np.zeros(shape=(self.Pdata.shape[0], self.Pdata.shape[1]*3+1)))
        self.Sr = pd.DataFrame(np.zeros(shape=(self.Pdata.shape[0], self.Pdata.shape[1]*3+1)))


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



def AcceClbNPosAuto(xacce_vec,yacce_vec,zacce_vec, herz, design_matrix=None,auto_design_matrix=False):
    """
    --> six position method to acceleration calibration
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
    # Ynorm = A[]
    AAA = A.T @(np.linalg.inv(A @ A.T)) 
    # 根据A矩阵截取片段
    xd = isEqualorNegative(xacce_vec, G, 0.01, 4*herz)
    yd = isEqualorNegative(yacce_vec, G, 0.01, 4*herz)
    zd = isEqualorNegative(zacce_vec, G, 0.01, 4*herz)
    L = np.array([]) # 3 * 6 
    # 根据A矩阵判断所需数据
    for i in range(A.shape[1]):
        for j in range(3):
            if A[j][i] == G:
                key = 'EqualDuration'
                if j == 0:
                    l=np.array([durationMean(xacce_vec, xd[key]), durationMean(yacce_vec, xd[key]),durationMean(zacce_vec, xd[key])])
                if j == 1:
                    l=np.array([durationMean(xacce_vec, yd[key]), durationMean(yacce_vec, yd[key]),durationMean(zacce_vec, yd[key])])
                if j == 2:
                    l=np.array([durationMean(xacce_vec, zd[key]), durationMean(yacce_vec, zd[key]),durationMean(zacce_vec, zd[key])])
            if A[j][i] == -G:
                key = 'NegativeDuration'
                if j == 0:
                    l=np.array([durationMean(xacce_vec, xd[key]), durationMean(yacce_vec, xd[key]),durationMean(zacce_vec, xd[key])])
                if j == 1:
                    l=np.array([durationMean(xacce_vec, yd[key]), durationMean(yacce_vec, yd[key]),durationMean(zacce_vec, yd[key])])
                if j == 2:
                    l=np.array([durationMean(xacce_vec, zd[key]), durationMean(yacce_vec, zd[key]),durationMean(zacce_vec, zd[key])])
            L = np.concatenate(L, l.T, axis=1)
    # 4x3 matrtix 
    return L @ A 




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
    return result


def continuousDuration(vec, value, threshold):
    result = []
    p1 = 0
    p2 = 0
    status = 0
    for i in range(len(vec)):
        if vec[i] == value:
            if p1 == 0:
                p1 = i
                p2 = i
            else:
                p2 = i
                if p2 - p1 >= threshold:
                    status = 1
        else:
            if status == 1:
                result.append((p1,p2))
            p1 = 0
            p2 = 0
            status = 0
    return result

def durationMean(vec, tuplelist):
    vv = np.array([])
    for each in tuplelist:
        vv = np.append(vv, vec[tuplelist[0]:tuplelist[1]])
    return np.mean(vv.flatten())
