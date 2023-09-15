import time
import numpy as np

import scipy.optimize as opt
import scipy.linalg as la
import scipy.stats as stats

class Model:
    def __init__(self, g, pi, alpha=0.01, K=6.5, T0=6, family="nbinom"):
        self._g = np.array(g)
        self._pi = np.array(pi)
        
        self.alpha = alpha
        self.K = K
        
        self.T = None
        self.T0 = T0
        
        self.family = family
                
    def GPi_setting(self, T):
        if self.T == T:
            return
        
        self.T = T
        
        self.pi = np.zeros(T+self.T0)
        self.pi[:min(self._pi.shape[0],T+self.T0)] = \
        self._pi[:min(self._pi.shape[0],T+self.T0)]
        self.pi = self.pi

        self.Pi = self.alpha * np.concatenate([[0],self.pi])[np.maximum(
            0, np.arange(T)[:,None] - np.arange(T)
        )]
        self.Pi0 = self.alpha * np.concatenate([[0],self.pi])[np.maximum(
            0, np.arange(T)[:,None] - np.arange(-self.T0,0)
        )]
        
        self.g = np.zeros(T+self.T0)
        self.g[:min(self._g.shape[0],T+self.T0)] = \
        self._g[:min(self._g.shape[0],T+self.T0)]

        self.G = np.concatenate([[0],self.g])[np.maximum(
            0, np.arange(T)[:,None] - np.arange(T)
        )]
        self.G0 = np.concatenate([[0],self.g])[np.maximum(
            0, np.arange(T)[:,None] - np.arange(-self.T0,0)
        )]
    
    def predict_R(self, A, beta):
        # ndarray
        A = np.array(A)
        beta = np.array(beta)
        
        assert(len(A.shape) == 2)
        assert(len(beta.shape) == 1)
        
        # parameter setting
        self.GPi_setting(A.shape[0])
        d = A.shape[1]
        
        assert(beta.shape[0] == d)
        
        # prediction
        o = np.exp(- A @ beta)
        p = 1 / (1 + o)
        q = o / (1 + o)

        R = self.K * p
        
        return R
        
    def predict_I(self, A, mu, beta):
        I0 = np.full(self.T0, np.exp(mu))
        R = self.predict_R(A, beta)
        L = R[:,None] * self.G
        L0 = R[:,None] * self.G0
        
        inv_ImL = la.inv(np.eye(self.T) - L)
        I = inv_ImL @ (L0 @ I0)
        
        return I
    
    def predict_EY(self, A, mu, beta):
        I0 = np.full(self.T0, np.exp(mu))
        I = self.predict_I(A, mu, beta)
        EY = self.Pi @ I + self.Pi0 @ I0
        
        return EY
    
    def ll(self, A, Y, nu, mu=None, beta=None, EY=None):
        # ndarray
        A = np.array(A)
        Y = np.array(Y)
        EY = np.array(EY)
        
        # parameter check
        assert(len(Y.shape) == 1)
        assert(len(A.shape) == 2)
        assert(A.shape[0] == Y.shape[0])
        assert(A.shape[0] == EY.shape[0])
        
        if EY is None:
            EY = self.predict_EY(A, mu, beta)
        else:
            self.GPi_setting(A.shape[0])
        
        # calculate log-likelihood
        if self.family == 'normal':
            return - self.T * np.log(nu) - np.nansum((Y-EY)**2)/(2*nu**2)
        elif self.family == 'nbinom':
            return np.nansum([_NBLL(yt, mt, nu) for yt, mt in zip(Y, EY)])
        else:
            raise
            
    def initialize(self, A, Y):
        # ndarray
        A = np.array(A)
        Y = np.array(Y)
        
        # parameter check
        assert(len(Y.shape) == 1)
        assert(len(A.shape) == 2)
        assert(A.shape[0] == Y.shape[0])
        
        # parameter setting
        self.GPi_setting(A.shape[0])
        
        Tyi = np.ceil(np.sum(np.arange(len(self.pi)) * self.pi)).astype(int)
        Ty = self.T - Tyi
        
        Gy = np.concatenate([[0],self.g])[np.maximum(
            0, np.arange(Ty)[:,None] - np.arange(-Tyi, Ty)
        )]
        
        # initialization by least square
        Yy = Y[Tyi:]
        Xy = A[:Ty]*(Gy@Y[:,None])
        
        # initialize beta
        bhaty = np.linalg.lstsq(Xy, Yy)[0]
        binit = np.concatenate([[4*bhaty[0]/self.K-2], 4*bhaty[1:]/self.K])
        
        # initialize mu
        # minit = np.log(np.nanmean(Y[:Tyi])/self.alpha)
        log_err_ratio = np.log(Y / self.predict_EY(A, 0, binit))
        minit = np.nanmean(log_err_ratio[np.isfinite(log_err_ratio)])
        
        return minit, binit
    
    def fit(self, A, Y, minit=None, binit=None,
            n_iter=3000, history=False, verbose=False, step_size=1):
        # ndarray
        A = np.array(A)
        Y = np.array(Y)
        binit = np.array(binit)
        
        # parameter check
        assert(len(Y.shape) == 1)
        assert(len(A.shape) == 2)
        assert(A.shape[0] == Y.shape[0])
        
        # parameter setting
        self.GPi_setting(A.shape[0])
        d = A.shape[1]
        
        # initialization
        if minit is None or binit is None:
            minit, binit = self.initialize(A, Y)
        
        assert(len(binit.shape) == 1)
        assert(binit.shape[0] == d)
        
        mhat = minit
        bhat = binit
        Ihat0 = np.full(self.T0, np.exp(mhat))

        if history:
            nhs = np.zeros(n_iter)
            mhs = np.zeros(n_iter)
            bhs = np.zeros([n_iter, d])
            lls = np.zeros(n_iter)
        
        # prediction
        o = np.exp(- A @ bhat)
        p = 1 / (1 + o)
        q = o / (1 + o)

        Rhat = self.K * p
        d1Rh = self.K * (p * q) * A.T
        d2Rh = self.K * (p * q * (q - p)) * (A.T[:,None,:] * A.T)

        Lhat = Rhat[:,None] * self.G
        Lhat0 = Rhat[:,None] * self.G0

        inv_ImLh = la.inv(np.eye(self.T)-Lhat) 
        Pi_inv_ImLh = self.Pi @ inv_ImLh

        Ihat = inv_ImLh @ (Lhat0 @ Ihat0)
        EYhat = self.Pi @ Ihat + self.Pi0 @ Ihat0
        
        # optimize for nu
        nhat = self.update_nu(Y, EYhat)

        # loglikelihood
        if verbose:
            print("Before EM, ll: %f"%self.ll(A, Y, nhat, EY=EYhat))

        for i in np.arange(n_iter):
            start_iter = time.time()

            # derivatives
            d1L = d1Rh[...,None] * self.G
            d1L0 = d1Rh[...,None] * self.G0
            
            dI_db = inv_ImLh @ (d1L @ Ihat[:,None] + d1L0 @ Ihat0[:,None])

            d2L = d2Rh[...,None] * self.G
            d2L0 = d2Rh[...,None] * self.G0

            dEY_dm = EYhat
            dEY_db = (self.Pi @ dI_db)[...,0]
            d2EY_dm2 = dEY_dm
            d2EY_dmdb = dEY_db
            d2EY_db2 = (Pi_inv_ImLh @ (
                d1L[:,None] @ dI_db + d1L @ dI_db[:,None]
                + d2L @ Ihat[:,None] + d2L0 @ Ihat0[:,None]))[...,0]
            
            if self.family == 'normal':
                dl_dEY = (Y - EYhat) / nhat**2
                d2l_dEY2 = np.full(self.T, -1/nhat**2)
            elif self.family == 'nbinom':
                dl_dEY = Y/EYhat - (nhat+Y)/(nhat+EYhat)
                d2l_dEY2 = - Y/EYhat**2 + (nhat+Y)/(nhat+EYhat)**2
            else:
                raise
            
            # first derivative
            dl_dm = np.nansum(dl_dEY * dEY_dm)
            dl_db = np.nansum(dl_dEY * dEY_db, 1)

            # second deivative
            d2l_dm2 = np.nansum(d2l_dEY2 * dEY_dm**2) \
                    + np.nansum(dl_dEY * d2EY_dm2)
            d2l_dmdb = np.nansum(d2l_dEY2 * dEY_dm * dEY_db, 1) \
                     + np.nansum(dl_dEY * d2EY_dmdb, 1)
            d2l_db2 = np.nansum(d2l_dEY2 * dEY_db * dEY_db[:,None,:], 2) \
                    + np.nansum(dl_dEY * d2EY_db2, 2)

            dl_dmb = np.concatenate([[dl_dm], dl_db])
            d2l_dmb2 = np.concatenate([
                np.concatenate([[d2l_dm2], d2l_dmdb])[None,:],
                np.concatenate([d2l_dmdb[:,None], d2l_db2], 1)])

            # Newton's method
            u, v = la.eig(-d2l_dmb2)
            dmb = np.real(((v/np.maximum(u,np.max(u)/1000)) @ v.T) @ dl_dmb)
            ss = np.min([1, 1/np.sqrt(np.sum(dmb**2))])

            mhat = mhat + ss * dmb[0]
            bhat = bhat + ss * dmb[1:]
            Ihat0 = np.full(self.T0, np.exp(mhat))

            # prediction
            o = np.exp(- A @ bhat)
            p = 1 / (1 + o)
            q = o / (1 + o)

            Rhat = self.K * p
            d1Rh = self.K * (p * q) * A.T
            d2Rh = self.K * (p * q * (q - p)) * (A.T[:,None,:] * A.T)

            Lhat = Rhat[:,None] * self.G
            Lhat0 = Rhat[:,None] * self.G0

            inv_ImLh = la.inv(np.eye(self.T)-Lhat) 
            Pi_inv_ImLh = self.Pi @ inv_ImLh

            Ihat = inv_ImLh @ (Lhat0 @ Ihat0)
            EYhat = self.Pi @ Ihat + self.Pi0 @ Ihat0

            # optimize for nu
            nhat = self.update_nu(Y, EYhat)

            # loglikelihood
            ll = self.ll(A, Y, nhat, EY=EYhat)

            if history:
                nhs[i] = nhat
                mhs[i] = mhat
                bhs[i] = bhat
                lls[i] = ll

            if verbose:
                print("%d-th iteration finished, ll: %f, lapse: %.3fsec."
                      %(i+1, ll, time.time()-start_iter))

        if history:
            return nhs, mhs, bhs, lls
        else:
            return nhat, mhat, bhat, ll
        
    def update_nu(self, Y, EY):
        # ndarray
        Y = np.array(Y)
        EY = np.array(EY)
        
        # parameter check
        assert(len(Y.shape) == 1)
        assert(len(EY.shape) == 1)
        assert(Y.shape[0] == EY.shape[0])
        
        # update nu
        if self.family == 'nbinom':
            return np.exp(opt.minimize_scalar(
                lambda x: -np.nansum([_NBLL(yt, mt, np.exp(x)) 
                                   for yt, mt in zip(Y, EY)])).x)
        elif self.family == 'normal':
            return np.sqrt(np.nanmean((Y-EY)**2))
        else:
            raise 
        
    def inference(self, A, Y, nu, mu, beta, L_HAC = None):
        # ndarray
        A = np.array(A)
        Y = np.array(Y)
        beta = np.array(beta)
        
        # parameter check
        assert(len(Y.shape) == 1)
        assert(len(A.shape) == 2)
        assert(A.shape[0] == Y.shape[0])
        
        # parameter setting
        self.GPi_setting(A.shape[0])
        d = A.shape[1]
        
        assert(len(beta.shape) == 1)
        assert(beta.shape[0] == d)

        I0 = np.full(self.T0, np.exp(mu))
        
        # prediction
        o = np.exp(- A @ beta)
        p = 1 / (1 + o)
        q = o / (1 + o)

        R = self.K * p
        d1R = self.K * (p * q) * A.T
        d2R = self.K * (p * q * (q - p)) * (A.T[:,None,:] * A.T)

        L = R[:,None] * self.G
        L0 = R[:,None] * self.G0

        inv_ImL = la.inv(np.eye(self.T) - L) 
        Pi_inv_ImL = self.Pi @ inv_ImL

        I = inv_ImL @ (L0 @ I0)
        EY = self.Pi @ I + self.Pi0 @ I0
        
        # derivatives
        d1L = d1R[...,None] * self.G
        d1L0 = d1R[...,None] * self.G0

        dI_db = inv_ImL @ (d1L @ I[:,None] + d1L0 @ I0[:,None])

        d2L = d2R[...,None] * self.G
        d2L0 = d2R[...,None] * self.G0

        dEY_dm = EY
        dEY_db = (self.Pi @ dI_db)[...,0]
        d2EY_dm2 = dEY_dm
        d2EY_dmdb = dEY_db
        d2EY_db2 = (Pi_inv_ImL @ (
            d1L[:,None] @ dI_db + d1L @ dI_db[:,None]
            + d2L @ I[:,None] + d2L0 @ I0[:,None]))[...,0]

        if self.family == 'normal':
            dlt_dn = np.array((Y - EY)**2 / nu**3 - 1 / nu)
            # dl_dn = np.sum((Y - EY)**2 / nu**3 - 1 / nu)
            dl_dEY = (Y - EY) / nu**2
            d2l_dn2 = np.sum(- 3 * (Y - EY)**2 / nu**4 + 1 / nu**2)
            d2l_dndEY = - 2 * (Y - EY) / nu**3
            d2l_dEY2 = np.full(self.T, -1/nu**2)
        elif self.family == 'nbinom':
            dlt_dn = np.array([_dNBLL_dr(yt, mt, nu) for yt, mt in zip(Y, EY)])
            # dl_dn = np.sum([_dNBLL_dr(yt, mt, nu) for yt, mt in zip(Y, EY)])
            dl_dEY = Y/EY - (nu+Y)/(nu+EY)
            d2l_dn2 = np.sum([_d2NBLL_dr2(yt, mt, nu) for yt, mt in zip(Y, EY)])
            d2l_dndEY = - 1/(nu+EY) + (nu+Y)/(nu+EY)**2
            d2l_dEY2 = - Y/EY**2 + (nu+Y)/(nu+EY)**2
        else:
            raise

        # Fisher information
        d2l_dndm = np.sum(d2l_dndEY * dEY_dm)
        d2l_dndb = np.sum(d2l_dndEY * dEY_db, 1)
        d2l_dm2 = np.sum(d2l_dEY2 * dEY_dm**2) \
                + np.sum(dl_dEY * d2EY_dm2)
        d2l_dmdb = np.sum(d2l_dEY2 * dEY_dm * dEY_db, 1) \
                 + np.sum(dl_dEY * d2EY_dmdb, 1)
        d2l_db2 = np.sum(d2l_dEY2 * dEY_db * dEY_db[:,None,:], 2) \
                + np.sum(dl_dEY * d2EY_db2, 2)
        
        d2l_dnm2 = np.array([[d2l_dn2, d2l_dndm],
                             [d2l_dndm, d2l_dm2]])
        d2l_dnmdb = np.stack([d2l_dndb, 
                              d2l_dmdb])
        d2l_dth2 = np.block([[d2l_dnm2, d2l_dnmdb],
                             [d2l_dnmdb.T, d2l_db2]])
        
        Ihat_dth = -d2l_dth2
        
        # HAC estimator
        dlt_dm = dl_dEY * dEY_dm
        dlt_db = dl_dEY * dEY_db
        dlt_dth = np.concatenate([[dlt_dn, dlt_dm], dlt_db], 0)
        
        if L_HAC is None:
            L_HAC = np.floor(4*(self.T/100)**(2/9))
        w_HAC = np.maximum(1 - np.abs(
            np.arange(self.T)[:,None] - np.arange(self.T))/L_HAC, 0)
        
        Ihat_hac = dlt_dth @ w_HAC @ dlt_dth.T
        
        # Sandwich estimator
        Ihat_sdw = Ihat_dth @ la.pinv(Ihat_hac) @ Ihat_dth
        
        return Ihat_dth, Ihat_hac, Ihat_sdw
    
    def confidence_R(self, A, beta, cov, cv, verbose=False):
         # ndarray
        A = np.array(A)
        beta = np.array(beta)
        cov = np.array(cov)
        
        # parameter check
        assert(len(A.shape) == 2)
        
        # parameter setting
        self.GPi_setting(A.shape[0])
        d = A.shape[1]
        
        assert(len(beta.shape) == 1)
        assert(beta.shape[0] == d)
        assert(len(cov.shape) == 2)
        assert(cov.shape == (d,d))
                
        up = beta + cv * np.sqrt(np.diag(cov))
        lo = beta - cv * np.sqrt(np.diag(cov))
        
        cons = lambda x: cv**2 - (beta - x) @ la.pinv(cov) @ (beta - x)
        
        conf_band = np.zeros([self.T, 2])
        beta_min = np.zeros([self.T, d])
        beta_max = np.zeros([self.T, d])
        for t in np.arange(self.T):
            start_iter = time.time()
            
            if t == 0:
                init_min = beta
                init_max = beta
            else:
                init_min = beta_min[t-1]
                init_max = beta_max[t-1]

            result_min = opt.minimize(
                lambda x: A[t] @ x, 
                init_min,
                bounds=np.transpose([lo, up]),
                constraints=opt.NonlinearConstraint(
                    cons, 0, cv**2, keep_feasible=False),
                hess = lambda x: np.zeros((d,d)),
                method='trust-constr'
            )
            result_max = opt.minimize(
                lambda x: - A[t] @ x, 
                init_max,
                bounds=np.transpose([lo, up]),
                constraints=opt.NonlinearConstraint(
                    cons, 0, cv**2, keep_feasible=False),
                hess = lambda x: np.zeros((d,d)),
                method='trust-constr'
            )
            conf_band[t] = np.array([self.K/(1+np.exp(-result_min.fun)), 
                                     self.K/(1+np.exp(+result_max.fun))])
            beta_min[t] = result_min.x
            beta_max[t] = result_max.x
            
            if verbose:
                print("optimization at t = %d finished, lapse: %.3fsec."
                      %(t+1, time.time()-start_iter))
            
        return conf_band, beta_min, beta_max
    
    def confidence_I(self, A, mu, beta, cov, cv, verbose=False):
        # ndarray
        A = np.array(A)
        beta = np.array(beta)
        
        # parameter check
        assert(len(A.shape) == 2)
        
        # parameter setting
        self.GPi_setting(A.shape[0])
        d = A.shape[1]
        
        assert(len(beta.shape) == 1)
        assert(beta.shape[0] == d)
        assert(len(cov.shape) == 2)
        assert(cov.shape == (d+1,d+1))
        
        theta = np.concatenate([[mu], beta])
        
        up = theta + cv * np.sqrt(np.diag(cov))
        lo = theta - cv * np.sqrt(np.diag(cov))
        
        cons = lambda x: cv**2 - (theta - x) @ la.pinv(cov) @ (theta - x)
        
        conf_band = np.zeros([self.T, 2])
        theta_min = np.zeros([self.T, d+1])
        theta_max = np.zeros([self.T, d+1])
        for t in np.arange(self.T):
            start_iter = time.time()
            
            if t == 0:
                init_min = theta
                init_max = theta
            else:
                init_min = theta_min[t-1]
                init_max = theta_max[t-1]

            result_min = opt.minimize(
                lambda x: self.predict_I(A[:t+1], x[0], x[1:])[-1], 
                init_min,
                bounds=np.transpose([lo, up]),
                constraints=opt.NonlinearConstraint(
                    cons, 0, cv**2, keep_feasible=False),
                method='trust-constr'
            )
            result_max = opt.minimize(
                lambda x: -self.predict_I(A[:t+1], x[0], x[1:])[-1], 
                init_max,
                bounds=np.transpose([lo, up]),
                constraints=opt.NonlinearConstraint(
                    cons, 0, cv**2, keep_feasible=False),
                method='trust-constr'
            )
            conf_band[t] = np.array([result_min.fun, -result_max.fun])
            theta_min[t] = result_min.x
            theta_max[t] = result_max.x
            
            if verbose:
                print("optimization at t = %d finished, lapse: %.3fsec."
                      %(t+1, time.time()-start_iter))
            
        return conf_band, theta_min, theta_max
        
    def confidence_EY(self, A, mu, beta, cov, cv, verbose=False):
        # ndarray
        A = np.array(A)
        beta = np.array(beta)
        
        # parameter check
        assert(len(A.shape) == 2)
        
        # parameter setting
        self.GPi_setting(A.shape[0])
        d = A.shape[1]
        
        assert(len(beta.shape) == 1)
        assert(beta.shape[0] == d)
        
        theta = np.concatenate([[mu], beta])
        
        up = theta + cv * np.sqrt(np.diag(cov))
        lo = theta - cv * np.sqrt(np.diag(cov))
        
        cons = lambda x: cv**2 - (theta - x) @ la.pinv(cov) @ (theta - x)
        
        conf_band = np.zeros([self.T, 2])
        theta_min = np.zeros([self.T, d+1])
        theta_max = np.zeros([self.T, d+1])
        for t in np.arange(self.T):
            start_iter = time.time()
            
            if t == 0:
                init_min = theta
                init_max = theta
            else:
                init_min = theta_min[t-1]
                init_max = theta_max[t-1]

            result_min = opt.minimize(
                lambda x: self.predict_EY(A[:t+1], x[0], x[1:])[-1], 
                init_min,
                bounds=np.transpose([lo, up]),
                constraints=opt.NonlinearConstraint(
                    cons, 0, cv**2, keep_feasible=False),
                method='trust-constr'
            )
            result_max = opt.minimize(
                lambda x: -self.predict_EY(A[:t+1], x[0], x[1:])[-1], 
                init_max,
                bounds=np.transpose([lo, up]),
                constraints=opt.NonlinearConstraint(
                    cons, 0, cv**2, keep_feasible=False),
                method='trust-constr'
            )
            conf_band[t] = np.array([result_min.fun, -result_max.fun])
            theta_min[t] = result_min.x
            theta_max[t] = result_max.x
            
            if verbose:
                print("optimization at t = %d finished, lapse: %.3fsec."
                      %(t+1, time.time()-start_iter))
            
        return conf_band, theta_min, theta_max
    
    def confidence_Y(self, A, nu, mu, beta, alpha, 
                     cov=None, cv=None, conf_EY=None, verbose=False):
        # ndarray
        A = np.array(A)
        beta = np.array(beta)
        
        # parameter check
        assert(len(A.shape) == 2)
        
        # parameter setting
        self.GPi_setting(A.shape[0])
        d = A.shape[1]
        
        assert(len(beta.shape) == 1)
        assert(beta.shape[0] == d)
        
        
        # get confidence band for EY
        conf_band = np.zeros([self.T, 2])
        if conf_EY is not None:
            conf_EY = np.array(conf_EY)
            assert(conf_EY.shape[0] == self.T)   
        elif (cov is not None) and (cv is not None):
            conf_EY, _, _ = self.confidence_EY(A, mu, beta, cov, cv, verbose)
        else:
            conf_EY = self.predict_EY(A, mu, beta)[:,None]
        
        # get confidence band for Y
        if self.family == 'nbinom':
                conf_band[:,0] = stats.nbinom.ppf(alpha/2, nu,
                                                  nu/(nu+conf_EY[:,0]))
                conf_band[:,1] = stats.nbinom.ppf(1-alpha/2, nu,
                                                  nu/(nu+conf_EY[:,-1]))
        elif self.family == 'normal':
            conf_band[:,0] = (conf_EY[:,0] 
                              + stats.norm.ppf(alpha/2) * nu)

            conf_band[:,1] = (conf_EY[:,-1]
                              + stats.norm.ppf(1-alpha/2) * nu)
        else:
            raise
        
            
        return conf_band
            
def _NBLL(y, m, r):
    if np.isfinite(y):
        y = int(y)
        return (np.sum(np.log(1+(r-1)/(np.arange(y)+1))) 
                + r * np.log(r/(r+m)) 
                + y * np.log(m/(r+m)))
    else:
        return np.nan

def _dNBLL_dr(y, m, r):
    if np.isfinite(y):
        y = int(y)
        return (np.sum(1/(np.arange(y)+r)) 
                + np.log(r/(r+m)) + (m-y)/(r+m))
    else:
        return np.nan

def _d2NBLL_dr2(y, m, r):
    if np.isfinite(y):
        y = int(y)
        return (-np.sum(1/(np.arange(y)+r)**2) 
                + 1/r - 1/(r+m) + (m-y)/(r+m)**2)
    else:
        return nan