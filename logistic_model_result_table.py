import statsmodels.api as sm
import numpy as np
from statsmodels.tools.decorators import (cache_readonly,
                                          cached_value, cached_data)
from statsmodels.compat.python import lrange, lmap, lzip
from statsmodels.iolib.table import SimpleTable
from scipy import stats
class logistic_output_table:
    """
    method == 'newton'
    """
    def __init__(self,sklearn_model,exog,use_t=False):
        self.model = sklearn_model
        self.x = sm.add_constant(exog)
        self.nobs = exog.shape[0]
        self.params = np.append(self.model.intercept_,self.model.coef_[0])
        self.df_resid = self.x.shape[0]-self.x.shape[1]
        self.use_t=use_t

    def cdf(self, X):
        """
        The logistic cumulative distribution function

        Parameters
        ----------
        X : array_like
            `X` is the linear predictor of the logit model.  See notes.

        Returns
        -------
        1/(1 + exp(-X))

        Notes
        -----
        In the logit model,

        .. math:: \\Lambda\\left(x^{\\prime}\\beta\\right)=
                  \\text{Prob}\\left(Y=1|x\\right)=
                  \\frac{e^{x^{\\prime}\\beta}}{1+e^{x^{\\prime}\\beta}}
        """
        X = np.asarray(X)
        return 1/(1+np.exp(-X))
    
    def hessian(self, params):
        """
        Logit model Hessian matrix of the log-likelihood

        Parameters
        ----------
        params : array_like
            The parameters of the model

        Returns
        -------
        hess : ndarray, (k_vars, k_vars)
            The Hessian, second derivative of loglikelihood function,
            evaluated at `params`

        Notes
        -----
        .. math:: \\frac{\\partial^{2}\\ln L}{\\partial\\beta\\partial\\beta^{\\prime}}=-\\sum_{i}\\Lambda_{i}\\left(1-\\Lambda_{i}\\right)x_{i}x_{i}^{\\prime}
        """
        X = self.x
        L = self.cdf(np.dot(X,params))
        return -np.dot(L*(1-L)*X.T,X)

    
    def Hinv(self,params):
        hess = self.hessian(params) / self.nobs
        return np.linalg.inv(-hess) / self.nobs
    
    @cached_value
    def bse(self):
        """The standard errors of the parameter estimates."""
        bse_ = np.sqrt(np.diag(self.Hinv(self.params)))
        return bse_
    @cached_value
    def tvalues(self):
        """
        Return the t-statistic for a given parameter estimate.
        """
        return self.params / self.bse
    @cached_value
    def pvalues(self):
        """The two-tailed p values for the t-stats of the params."""
        #  t检验的p值，或者z检验的值，看样本量大小。默认为z检验
        if self.use_t:
            return stats.t.sf(np.abs(self.tvalues), self.df_resid) * 2
        else:
            return stats.norm.sf(np.abs(self.tvalues)) * 2
    @cached_value
    def conf_int(self, alpha=.05, cols=None):
        """
        Construct confidence interval for the fitted parameters.

        Parameters
        ----------
        alpha : float, optional
            The significance level for the confidence interval. The default
            `alpha` = .05 returns a 95% confidence interval.
        cols : array_like, optional
            Specifies which confidence intervals to return.

        Returns
        -------
        array_like
            Each row contains [lower, upper] limits of the confidence interval
            for the corresponding parameter. The first column contains all
            lower, the second column contains all upper limits.

        Notes
        -----
        The confidence interval is based on the standard normal distribution
        if self.use_t is False. If self.use_t is True, then uses a Student's t
        with self.df_resid_inference (or self.df_resid if df_resid_inference is
        not defined) degrees of freedom.
        """
        bse = self.bse

        if self.use_t:
            dist = stats.t
            df_resid = getattr(self, 'df_resid_inference', self.df_resid)
            q = dist.ppf(1 - alpha / 2, df_resid)
        else:
            dist = stats.norm
            q = dist.ppf(1 - alpha / 2)

        params = self.params
        lower = params - q * bse
        upper = params + q * bse
        return np.asarray(lzip(lower, upper))
    
    def forg(self,x, prec=3):
        if prec == 3:
        # for 3 decimals
            if (abs(x) >= 1e4) or (abs(x) < 1e-4):
                return '%9.3g' % x
            else:
                return '%9.3f' % x
        elif prec == 4:
            if (abs(x) >= 1e4) or (abs(x) < 1e-4):
                return '%10.4g' % x
            else:
                return '%10.4f' % x
        else:
            raise ValueError("`prec` argument must be either 3 or 4, not {prec}"
                         .format(prec=prec))
            
    def summary(self, xname=None,title='Summarize the Loistic Regression Results', alpha=.05):
        """Summarize the Regression Results
        """
        exog_idx = lrange(len(self.params))
        params = self.params
        std_err = self.bse
        tvalues = self.tvalues
        pvalues = self.pvalues
        conf_int = self.conf_int
        if self.use_t:
            param_header = ['coef', 'std err', 't', 'P>|t|',
                        '[' + str(alpha/2), str(1-alpha/2) + ']']
        else:
            param_header = ['coef', 'std err', 'z', 'P>|z|',
                        '[' + str(alpha/2), str(1-alpha/2) + ']']
        if xname is None:
            xname = ['x_%d' % i for i in range(len(self.params))]
            xname[0] = 'const'
        else:
            xname.insert(0,'const')
        if len(xname) != len(params):
            raise ValueError('xnames and params do not have the same length')
        params_stubs = xname
    
        params_data = lzip([self.forg(params[i],4) for i in exog_idx],
                       [self.forg(std_err[i]) for i in exog_idx],
                       [self.forg(tvalues[i]) for i in exog_idx],
                       [self.forg(pvalues[i]) for i in exog_idx],
                       [self.forg(conf_int[i,0]) for i in exog_idx],
                       [self.forg(conf_int[i,1]) for i in exog_idx])
        parameter_table = SimpleTable(params_data,
                                      param_header,
                                      params_stubs,
                                  title=title
                                  )

        return parameter_table
    
    