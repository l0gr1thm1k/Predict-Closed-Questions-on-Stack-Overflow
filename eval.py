# author: Dmitry Kan dmitry.kan@gmail.com
import scipy as sp
import numpy as np

debug = 0
verbose = 0

# Multi Class Log Loss
# both obs_act and obs_pred parameters are arrays of floats
# the obs_pred contains exactly one 1, the rest are zeros
def mcllfun_single_obs_pair(obs_act,obs_pred):

    if debug:
        print obs_act
        print obs_pred
    
    if (len(obs_act) != len(obs_pred)):
        raise Exception("Actual observation and predicted observation are of different lengths!")
    
    # the number of classes
    m = len(obs_act)
    # epsilon for adjustments
    epsilon = 1e-15
    
    pred_sum=0.0
    for p_pred in obs_pred:
        pred_sum = pred_sum+p_pred
    if debug:
        print "pred_sum=" + str(pred_sum)
    
    if pred_sum == 0:
        pred_sum = 1
    
    for i in range(0,m):
        # adjust
        obs_pred[i] = sp.maximum(sp.minimum(obs_pred[i],1-epsilon),epsilon)
        # rescale
        obs_pred[i] = obs_pred[i] / pred_sum

    if debug:    
        print "After rescaling, obs_pred=" + str(obs_pred)
    
    ll = 0.0
    for j in range(0,m):
        if obs_pred[j] > 0:
            ll = ll + obs_act[j] * sp.log(obs_pred[j])
    return ll
    
# params are matrices, where each row is an observation
# where indices are organized like this:
# [row][col]
def mcllfun(probs_act,probs_pred):
    # return value of MultiClass LogLoss
    mcll = 0.0
    # number of observations
    n = len(probs_act)
    for i in range(0,n):
        if debug:
            print i
            print probs_act[i]
            print probs_pred[i]
        mcll = mcll + mcllfun_single_obs_pair(probs_act[i], probs_pred[i])
    
    if verbose:
        print "n=" + str(n)
        print "mcll=" + str(mcll)
    
    return -1.0 * (1.0/n) * mcll

def multiclass_log_loss(y_true, y_pred, eps=1e-15):
    """Multi class version of Logarithmic Loss metric.
    https://www.kaggle.com/wiki/MultiClassLogLoss

    idea from this post:
    http://www.kaggle.com/c/emc-data-science/forums/t/2149/is-anyone-noticing-difference-betwen-validation-and-leaderboard-error/12209#post12209

    Parameters
    ----------
    y_true : array, shape = [n_samples]
    y_pred : array, shape = [n_samples, n_classes]

    Returns
    -------
    loss : float
    """
    predictions = np.clip(y_pred, eps, 1 - eps)

    # normalize row sums to 1
    predictions /= predictions.sum(axis=1)[:, np.newaxis]

    actual = np.zeros(y_pred.shape)
    rows = actual.shape[0]
    actual[np.arange(rows), y_true.astype(int)] = 1
    vsota = np.sum(actual * np.log(predictions))
    return -1.0 / rows * vsota
    
def main():
    _act = [ [1.0, 0.0],
             [0.0, 1.0],
             [1.0, 0.0]
           ]
    # prediction is quite far, mcll=0.657561671274
    #_pred = [ [0.011327211985872214, 0.021327211985872214],
    #          [0.012327211985872214, 0.031327211985872214],
    #          [0.052327211985872214, 0.041327211985872214]
    #        ]

    # prediction is quite close, mcll=0.0262542903368
    _pred = [ [0.911327211985872214, 0.021327211985872214],
              [0.012327211985872214, 0.931327211985872214],
              [0.952327211985872214, 0.041327211985872214]
            ]
    print "mcll=" + str(mcllfun(_act,_pred))
    
    _act_np = np.array([0,1,2])
    _pred_np = np.array([[1,0,0],[0,1,0],[0,0,1]])
    print "mcll_forum_version=" + str(multiclass_log_loss(_act_np,_pred_np))
    
    _act = [ [1, 0, 0],
             [0, 1, 0],
             [0, 0, 1]
           ]
    _pred = [[1,0,0],[0,1,0],[0,0,1]]
    print "mcll=" + str(mcllfun(_act,_pred))
    
if __name__=="__main__":
    main()