import numpy as np

import scipy.optimize as opt
import scipy.linalg as la
import scipy.stats as stats

class CV:
    tol = 1e-12
    
    def __init__(self, d, num_mc = 10000, random_seed = 230113):
        self.d = d
        np.random.seed(seed=random_seed)
        
        if type(d) != int or d <= 0:
            raise "Dimension 'd' (%f) is not a positive integer"%d
        elif d == 1:
            self.num_mc = 1
            self.chi2_mc = np.zeros([1])
        else:
            self.num_mc = num_mc
            self.chi2_mc = stats.chi2(d-1).rvs(num_mc)
        return
        
    def r_0(self, u, chi):
        u = np.array(u).astype(float)
        u = u.reshape(u.shape+(1,)*np.array(chi).ndim)

        idx = (np.sqrt(u)- chi) > 5
        return np.where(idx, 1,
            stats.norm.cdf(-np.sqrt(u)-chi)
            + stats.norm.cdf(np.sqrt(u)-chi))
    
    def d1r_0(self, u, chi):
        u = np.array(u).astype(float)
        u = u.reshape(u.shape+(1,)*np.array(chi).ndim)

        idx = u < 1e-8
        return np.where(idx, stats.norm.pdf(chi)*chi,
            (stats.norm.pdf(np.sqrt(u)-chi)
             - stats.norm.pdf(np.sqrt(u)+chi))
            / (2 * np.sqrt(np.maximum(u, 1e-8))))
    
    def d2r_0(self, u, chi):
        u = np.array(u).astype(float)
        u = u.reshape(u.shape+(1,)*np.array(chi).ndim)

        idx = u < 2e-6
        return np.where(idx, stats.norm.pdf(chi)*chi*(chi**2-3)/6,
            (stats.norm.pdf(np.sqrt(u)+chi)*(chi*np.sqrt(u)+u+1)
             + stats.norm.pdf(np.sqrt(u)-chi)*(chi*np.sqrt(u)-u-1))
            / (4 * np.sqrt(np.maximum(u, 2e-6)**3)))
    
    def d3r_0(self, u, chi):
        u = np.array(u).astype(float)
        u = u.reshape(u.shape+(1,)*np.array(chi).ndim)

        idx = u < 2e-4
        return np.where(idx, stats.norm.pdf(chi)*(chi**5-10*chi**3+15*chi)/60,
            (stats.norm.pdf(chi-np.sqrt(u))
             *(u**2-2*chi*np.sqrt(u**3)+(2+chi**2)*u-3*chi*np.sqrt(u)+3)
             - stats.norm.pdf(chi+np.sqrt(u))
             *(u**2+2*chi*np.sqrt(u**3)+(2+chi**2)*u+3*chi*np.sqrt(u)+3))
            / (8 * np.sqrt(np.maximum(u, 2e-4)**5)))
    
    def r_d(self, u, chi):
        chi_0 = np.sqrt(np.maximum(np.array(chi)[...,None]**2 
                                   - self.chi2_mc, 0))
        return np.mean(self.r_0(u, chi_0), -1)
    
    def d1r_d(self, u, chi):
        chi_0 = np.sqrt(np.maximum(np.array(chi)[...,None]**2 
                                   - self.chi2_mc, 0))
        return np.mean(self.d1r_0(u, chi_0), -1)
    
    def d2r_d(self, u, chi):
        chi_0 = np.sqrt(np.maximum(np.array(chi)[...,None]**2 
                                   - self.chi2_mc, 0))
        return np.mean(self.d2r_0(u, chi_0), -1)
    
    def d3r_d(self, u, chi):
        chi_0 = np.sqrt(np.maximum(np.array(chi)[...,None]**2 
                                   - self.chi2_mc, 0))
        return np.mean(self.d3r_0(u, chi_0), -1)
    
    def ip_u0(self, chi):
        if chi**2 < 3:
            return 0, 0
        else:
            f = lambda x: (self.r_0(x, chi) - x*self.d1r_0(x, chi) 
                           - self.r_0(0, chi))

            if (np.abs(self.d2r_0(chi**2 - 3/2, chi)) < self.tol 
                or chi**2-chi**2-3 == 0):
                ip = chi**2 - 3/2
            else:
                ip = opt.brentq(lambda x: self.d2r_0(x, chi), 
                                chi**2-3, chi**2, xtol=self.tol)

            up = 2 * chi**2; lo = ip
            while f(up) < 0:
                lo = up; up = 2*up

            u0 = lo
            if f(lo) < 0:
                u0 = opt.brentq(f, lo, up, xtol=self.tol)
            elif f(lo) > self.tol:
                warnings.warn("Failed to solve for u0 at chi=%f"%chi)
        return ip, u0
    
    def rt_d(self, x, u, chi):
        x = np.sort(x)
        assert x[0] <= u and u <= x[1]
        return np.where(x[1]-x[0] < 1e-8, self.r_d(u,chi),
            ((x[1]-u) * self.r_d(x[0],chi) + (u-x[0]) * self.r_d(x[1],chi))
            / np.maximum(x[1]-x[0],1e-8))
    
    def delta(self, u, u1, chi):
        return np.where(np.abs(u-u1) < 1e-4, self.d2r_d(u1,chi)/2,
            (self.r_d(u,chi) - self.r_d(u1,chi) 
             - self.d1r_d(u1,chi)*(u-u1))
            / np.maximum(np.abs(u-u1),1e-4)**2)
    
    def d1delta(self, u, u1, chi):
        return np.where(np.abs(u-u1) < 1e-3, self.d3r_d(u1,chi)/6,
            (self.d1r_d(u,chi) + self.d1r_d(u1,chi) 
             - 2 * (self.r_d(u,chi) - self.r_d(u1,chi))
                 * np.sign(u-u1) / np.maximum(np.abs(u-u1),1e-3))
            / np.maximum(np.abs(u-u1),1e-3)**2)
    
    def lam(self, u1, chi):
        ip, u0 = self.ip_u0(chi)
        if u1 >= ip:
            us = np.array([0, u0])
        else:
            # print(chi, u1, ip, u0)
            us = np.sort([0, u1, ip, u0])

        dus = self.delta(us, u1, chi)
        d1dus = self.d1delta(us, u1, chi)

#         if np.all(d1dus <= 0) and np.argmax(dus)==0:
#             return {'val': dus[0], 'u2': 0}
#         elif np.all(np.diff((d1dus>=0).astype(int))<=0) and d1dus[-1]<=0:
#             idx = np.maximum(np.argmin(d1dus > 0), 1)
#             lo = us[idx-1]; up = us[idx]
#         elif np.min(np.abs(d1dus)) < 1e-6:
#             idx = np.argmax(dus)
#             lo = us[np.maximum(idx-1,0)]
#             up = us[np.minimum(idx+1,len(us)-1)]
#         else:
#             warnings.warn("""
#                 There are multiple optima in the function 
#                 delta(u, u1=%f, chi=%f)
#             """%(u1,chi))
#             lo = 0; up = u0

#         result_opt = opt.minimize_scalar(lambda x: -delta(x, u1, chi),
#                                          [lo, up], tol=tol)
        result_opt = opt.shgo(lambda x: -self.delta(x, u1, chi),
                              [(0, u0)], options={'f_tol':self.tol})

        if -result_opt.fun > dus[0]:
            return {'val': -result_opt.fun, 'u2': result_opt.x[0]}
        elif -result_opt.fun > dus[0] - 10**3*self.tol:
            return {'val': dus[0], 'u2': 0}
        else:
            warnings.warn("Optimum may be wrong for lam(u1=%f, chi=%f)"
                          %(u1,chi))
            return {'val': dus[0], 'u2': 0}
        
    def rho(self, chi, m2, m4=np.inf):
        _, u0 = self.ip_u0(chi)

        if m4 <= m2**2:
            return {'alpha': self.r_d(m2, chi), 
                    'u': np.array([0, m2]), 'p': np.array([0, 1])}
        elif m2 >= u0:
            return {'alpha': self.r_d(m2, chi), 
                    'u': np.array([0, m2]), 'p': np.array([0, 1])}
        elif not np.isfinite(m4):
            result_opt = opt.shgo(lambda u: -self.rt_d(u, m2, chi), 
                                  [(0, m2),(m2, u0)],
                                  options={'f_tol':self.tol})
            u = result_opt.x
            p1 = (u[1]-m2)/np.maximum(u[1]-u[0],1e-8)
            return {'alpha': -result_opt.fun, 
                    'u': u, 'p': np.array([p1, 1-p1])}
        else:
            ubar = self.lam(0, chi)['u2']
            lammax = lambda u1: (np.where(u1 >= ubar, self.delta(0, u1, chi),
                                      np.maximum(self.lam(u1, chi)['val'], 0)))
            obj = lambda u1: (self.r_d(u1[0], chi) 
                              + self.d1r_d(u1[0], chi)*(m2 - u1[0]) 
                              + lammax(u1[0])*(m4 + u1[0]**2 - 2*u1[0]*m2))

            result_opt = opt.shgo(obj, [(0, u0)], options={'f_tol':self.tol})
#             result_opt1 = opt.shgo(obj, [(0, ubar)], options={'f_tol':tol})
#             result_opt2 = opt.shgo(obj, [(ubar,u0)], options={'f_tol':tol})
#             result_opt1 = opt.minimize_scalar(obj, [0, ubar])
#             result_opt2 = opt.minimize_scalar(obj, [ubar,u0])
#             result_opt = np.where(result_opt1.fun < result_opt2.fun,
#                                   result_opt1, result_opt2)

            u1 = result_opt.x[0]
            u2 = self.lam(u1, chi)['u2']
            u = np.sort([u1, u2])
            p1 = (0 if u[1]-u[0] < 1e-8 
                  else (u[1] - m2)/np.maximum(u[1] - u[0], 1e-8))
            return {'alpha': result_opt.fun, 'u': u, 'p': np.array([p1, 1-p1])}
        
    def rho_l(self, chi, m2, m4=np.inf, num_us=5000):
        _, u0 = self.ip_u0(chi)
        
        us = np.sort(np.concatenate([[m2], np.linspace(0, 2*u0, num_us)]))
        result_linprog = opt.linprog(
            -self.r_d(us, chi), us[None,:]**2, [max(m4, m2**2+1e-4)], 
            np.stack([us**0, us], 0), [1, m2], [(0, 1) for _ in us])
        return{'alpha': -result_linprog.fun, 'u': us[result_linprog.x > 0],
               'p': result_linprog.x[result_linprog.x > 0]}
    
    def chi_pa(self, alpha):
        return np.sqrt(stats.chi2(self.d).ppf(1-alpha))
        
    def chi_eb(self, alpha, m2, m4=np.inf, linear=False):
        if not np.isfinite(m2):
            return {'cv':None, 'alpha':alpha, 'u':[0, m2], 'p':[0, 1]}
        elif 1/m2 < self.tol and np.isfinite(m4):
            warnings.warn("""
                Value of m2 (%f) is too large to reliably 
                compute the critical value.
                Assuming m4 constraint not binding.
            """)
            return self.cv_eb(alpha, m2)
        elif linear:
            lo = 0; up = np.sqrt((1+m2)/alpha)
#             if np.abs(self.rho_l(up, m2)['alpha'] - alpha) > 9e-6:
#                 up = opt.brentq(lambda chi: self.rho_l(chi, m2)['alpha'] - alpha, 
#                                 lo, up, xtol=self.tol)
            # if np.abs(self.rho_l(up, m2, m4)['alpha'] - alpha) > 1e-5:
            cv = opt.brentq(lambda chi: (self.rho_l(chi, m2, m4)['alpha'] 
                                         - alpha),
                            lo, up, xtol=self.tol)
            result = {'cv':cv}; result.update(self.rho_l(cv, m2, m4))
            return result
        else:
            lo = 0; up = np.sqrt((1+m2)/alpha)
#             if np.abs(self.rho(up, m2)['alpha'] - alpha) > 9e-6:
#                 up = opt.brentq(lambda chi: self.rho(chi, m2)['alpha'] - alpha, 
#                                 lo, up, xtol=self.tol)
#             if np.abs(self.rho(up, m2, m4)['alpha'] - alpha) > 1e-5:
            cv = opt.brentq(lambda chi: (self.rho(chi, m2, m4)['alpha'] 
                                         - alpha),
                            lo, up, xtol=self.tol)
            result = {'cv':cv}; result.update(self.rho(cv, m2, m4))
            return result