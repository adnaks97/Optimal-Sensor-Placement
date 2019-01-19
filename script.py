# imports
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
from sklearn.linear_model import Lasso
from sklearn.decomposition import pca
import seaborn as sns
from numpy.linalg import inv,pinv
from sklearn.linear_model import OrthogonalMatchingPursuit
from sklearn.linear_model import LinearRegression, LassoCV, Lasso, MultiTaskLassoCV, MultiTaskLasso
from sklearn.preprocessing import StandardScaler,MinMaxScaler
import scipy.io as sio
import PyQt4
from oct2py import Oct2Py,octave
sns.set_context()
octave.addpath("/home/skanda/Documents/Ericsson")
os.chdir('/home/skanda/Documents/Ericsson/SmartCities/newData')

# Recompute function
def recompute(A, iden = False):
    mat = np.zeros((A.shape[0], A.shape[1]), dtype=float)
    if iden is True:
        mat = A
    # print mat.shape
    N1, N2 = gen(N, X)
    # print N1.shape,N2.shape
    B = np.matmul(pinv(N1), N2)
    B = np.dot(-1, B)
    C = B.flatten('F')
    # print C.shape[0]
    nz_inds = Z.nonzero()[0]
    # print nz_inds
    for j in nz_inds:
        for i in range(Z.shape[0]):
            # print i,j,mat[i,j]
            if Z[i] == 0:
                # print mat[i,j]
                mat[i,j] = C[0]
                # print C[0]
                # print i,j
                C = C[1:]
            else:
                mat[i,i] = 1
                #print i,j
    return mat, B, C

# GEN function
def gen(N, X):
    z_ind = np.where(Z == 0)
    nz_ind = Z.nonzero()
    # print z_ind, nz_ind
    N1 = N[:, z_ind[0]]
    N2 = N[:, nz_ind[0]]
    return N1,N2

# Visualize the NORM VS N_NON_ZEROS FUNCTION
def viz(norms,start):
    sc1,sc2 = MinMaxScaler(), MinMaxScaler()
    comps = np.arange(start,11,1)
    costs = 300*comps
    norms = sc1.fit_transform(norms)
    costs = sc1.fit_transform(costs)
    fig, ax1 = plt.subplots()
    ax1.scatter(comps, norms[start-4:])
    ax1.plot(comps, norms[start-4:])
    ax1.set_xlabel("COMPONENTS")
    ax1.set_ylabel("NORM")
    ax2 = ax1.twinx()
    ax2.set_ylabel("COST")
    ax2.plot(comps, costs)
    plt.savefig('NewDatawithCost.png')
    plt.show()

# norms array
norms = np.ndarray(0, dtype = float)

# Reading the datset
for name in os.listdir('./'):
    #d = dict()
    df = pd.read_csv('Relative Humidity.csv')
    df.index = pd.to_datetime(df['TIMESTAMP'])
    df = df.drop(['TIMESTAMP'], axis =1)
    data = df.values
    #
    """data = sio.loadmat('Matrix.mat')
    data = data['Y']"""

    data

    # Applying PCA
    scX = MinMaxScaler()
    fogs = scX.fit_transform(data)
    U, Sigma, VT = np.linalg.svd(fogs)
    s_sum = Sigma.cumsum()
    Sigma, s_sum
    N_COMP = np.where(s_sum <= 0.9*s_sum[Sigma.shape[0]-1])
    y = np.where(s_sum > 0.9*s_sum[Sigma.shape[0]-1])
    y = y[0][1:]
    N = VT[:, y]
    N = N.T
    Y = fogs.T
    s_sum
    for s in s_sum:
        print float(s/s_sum[s_sum.shape[0]-1])

    iden = True
    A = np.eye(Y.shape[0],Y.shape[0])
    scaler = StandardScaler()
    Y = scaler.fit_transform(data.T)
    reg = Lasso(alpha=0.03, positive=True)
    reg.fit(A, Y)
    X = reg.coef_
    Z = np.count_nonzero(X, axis=0)
    vals = np.argsort(Z)
    Z
    vals[9-y.shape[0]:]
    d[name] = vals

d
vals+1
# Initialising A and T
for COMP in range(1,10):
    A = np.eye(Y.shape[0],Y.shape[0])
    T = np.zeros((Y.shape[0],Y.shape[0]),dtype = float)
    scaler = StandardScaler()
    Y = scaler.fit_transform(data.T)
    X = octave.OMP(A, Y, COMP)
    X = X.toarray()
    _X = np.copy(X)
    Z = np.count_nonzero(X, axis = 1)
    vals = np.argsort(Z)
    nZinds = vals[9-COMP:]
    Zinds = vals[:9-COMP]
    Z[nZinds] = 1
    Z[Zinds] = 0
    X[nZinds] = 1
    X[Zinds] = 0

    #Recompute A
    A,_,__ = recompute(A,iden)
    iden = False

    #Multiply and compute norm
    yhat = np.matmul(A,_X)
    scY = StandardScaler()
    yhat = scY.fit_transform(yhat)
    yhat[nZinds] = Y[nZinds]

    n1 = np.linalg.norm(Y- yhat)
    n2 = np.linalg.norm(Y)
    norm = n1/n2
    print 'NORM =',norm,'FOR',COMP,'NONZEROS'
    print Y[Zinds]
    print yhat[Zinds]
    print
    # break

    #Append norm value
    norms = np.append(norms,norm)

# Visualize the NORM VS N_NON_ZEROS

A
_X

viz(norms,4)
