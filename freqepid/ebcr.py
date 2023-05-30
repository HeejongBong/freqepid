import numpy as np
import numpy.linalg as la

import scipy.optimize as opt
import scipy.stats as stats

from freqepid import CV
    
class EBCR:
    tol = 1e-12
    
    def __init__(self, thats, covs, weights=None, **kwargs):
        # ndarray
        thats = np.array(thats)
        covs = np.array(covs)
        
        # parameter check
        assert(len(thats.shape) == 2)
        
        self.N = thats.shape[0]
        self.d = thats.shape[1]
        
        assert(covs.shape == (self.N, self.d, self.d))
        
        self.ths_pa = thats
        self.covs_pa = covs
        
        if weights is None:
            weights = np.array([
                np.exp(- la.slogdet(cov)[1]/self.d)
                for cov in covs
            ])
        else:
            weights = np.array(weights)
        
        assert(weights.shape == (self.N,))
        
        self.weights = weights
        
        self.th_o = np.sum(weights[:,None] * thats, 0) / np.sum(weights)
        self.eps = thats - self.th_o
        self.Phi2 = (np.sum(weights[:,None,None] 
                     * (self.eps[:,:,None] * self.eps[:,None,:] - covs), 0)
                     / np.sum(weights)) 
        
        self.Ws_eb = la.pinv(np.eye(self.d) + covs@la.pinv(self.Phi2))
        self.ths_eb = self.th_o + (self.Ws_eb @ self.eps[:,:,None])[...,0]
        self.covs_eb = self.Ws_eb @ covs @ self.Ws_eb.transpose([0,2,1])
        
        As = la.pinv(self.Phi2) @ covs @ la.pinv(self.Phi2)
        
        self.m2s = np.zeros(self.N)
        self.m4s = np.zeros(self.N)
        for i in np.arange(self.N):
            self.m2s[i], self.m4s[i] = self.moments(As[i])
        
        self.cv = CV(self.d, **kwargs)
        return
        
    def moments(self, A):
        trAYYt = np.sum(A * self.eps[:,:,None] * self.eps[:,None,:], axis=(1,2))
        trAS = np.sum(A * self.covs_pa, axis=(1,2))

        ASA = A @ self.covs_pa @ A
        trASAYYt = np.sum(ASA * self.eps[:,:,None] * self.eps[:,None,:],
                          axis=(1,2))
        trASAS = np.sum(ASA * self.covs_pa, axis=(1,2))

        m2 = max(0, np.sum(self.weights * (trAYYt - trAS)) 
                 / np.sum(self.weights))
        m4 = max(m2**2, np.sum(self.weights * (
                     (trAYYt - trAS)**2 + 2*trASAS - 4*trASAYYt)) 
                 / np.sum(self.weights))

        return m2, m4
    
    def chi_pa(self, alpha):
        return self.cv.chi_pa(alpha)
    
    def chi_eb(self, i, alpha, linear):
        return self.cv.chi_eb(alpha, self.m2s[i], self.m4s[i], linear)['cv']