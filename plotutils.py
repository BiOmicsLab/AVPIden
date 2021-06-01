from scipy import interpolate
import numpy as np


def roc_lininterp(x, y, length):
    """
    interporlation of the bootstrap roc_curves
    """    
    f = interpolate.interp1d(x, y, kind='slinear')
    xnew = np.linspace(0, 1, length)
    return xnew, f(xnew)


def mean_roc_kfold(curveitms):
    rocs = [[curve['fpr'], curve['tpr']] for curve in curveitms]
    minlen = min(len(roc[0]) for roc in rocs)
    for i in range(len(rocs)):
        rocs[i][0], rocs[i][1] = roc_lininterp(rocs[i][0], rocs[i][1], minlen)
    meanfpr = np.array([roc[0] for roc in rocs]).mean(axis=0)  # all fpr are regulized as same
    meantpr = np.array([roc[1] for roc in rocs]).mean(axis=0)
    stddtpr = np.array([roc[1] for roc in rocs]).std(axis=0)
    
    meantpr[0] = 0
    meanfpr[0] = 0
    stddtpr[0] = 0
    meantpr[-1] = 1
    stddtpr[-1] = 0
    meanfpr[-1] = 1
    
    aucs = np.asarray([curve['aucroc'] for curve in curveitms])
    meanauc = aucs.mean()
    stddauc = aucs.std()
    
    return meanfpr, meantpr, stddtpr, meanauc, stddauc