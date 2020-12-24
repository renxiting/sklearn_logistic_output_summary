import statsmodels.api as sm
import numpy as np
import pandas as pd
from statsmodels.tools.decorators import (cache_readonly,
                                          cached_value, cached_data)
from statsmodels.compat.python import lrange, lmap, lzip
from statsmodels.iolib.table import SimpleTable
from scipy import stats
from statsmodels.stats.contrast import (ContrastResults, WaldTestResults,
                                        t_test_pairwise)
from sklearn.linear_model import LogisticRegression

from datetime import datetime
import matplotlib.pyplot as plt
import time
from numpy import sum, logical_and, sqrt
from sklearn.metrics import roc_curve, auc
from sklearn.metrics import classification_report, accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn.utils.validation import column_or_1d

#%matplotlib inline
import seaborn as sns
pd.options.display.max_columns = 100
plt.rcParams["figure.figsize"] = (10,12)

from sklearn.metrics import roc_auc_score, roc_curve
from sklearn.externals import joblib
from sklearn.impute import SimpleImputer
#from sklearn2pmml.pipeline import PMMLPipeline
#from sklearn2pmml import sklearn2pmml
from sklearn.model_selection import train_test_split
from sklearn.feature_selection import SelectFromModel
from sklearn.externals.joblib import dump
from sklearn.externals.joblib import load

import warnings
warnings.filterwarnings("ignore")

class logistic_output_table:
    """
    method == 'newton'
    """
    def __init__(self,sklearn_model,exog,use_t=False):
        self.model = sklearn_model
        #self.x = sm.add_constant(exog)
        self.x = exog
        self.nobs = exog.shape[0]
        self.params = self.model.coef_[0]
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
    @cached_value
    def vif(self):
        from statsmodels.stats.outliers_influence import variance_inflation_factor
        col = list(range(self.x.shape[1]))
        vif = [variance_inflation_factor(self.x.iloc[:,col].values, ix)
               for ix in range(self.x.iloc[:,col].shape[1])]
        return vif
    
    def forg(self,x, prec=3):
        if prec == 3:
        # for 3 decimals
            if (abs(x) >= 1e4) or (abs(x) < 1e-4):
                return '%9.6g' % x
            else:
                return '%9.6f' % x
        elif prec == 4:
            if (abs(x) >= 1e4) or (abs(x) < 1e-4):
                return '%10.6g' % x
            else:
                return '%10.6f' % x
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
        try:
            vif = self.vif
        except:
            vif = np.ones(len(self.params))
        
        if self.use_t:
            param_header = ['coef', 'std err', 't', 'P>|t|',
                        '[' + str(alpha/2), str(1-alpha/2) + ']','vif']
        else:
            param_header = ['coef', 'std err', 'z', 'P>|z|',
                        '[' + str(alpha/2), str(1-alpha/2) + ']','vif']
        if xname is None:
            xname = ['x_%d' % i for i in range(len(self.params))]
            xname[0] = 'const'
        else:
            xname = xname
        if len(xname) != len(params):
            raise ValueError('xnames and params do not have the same length')
        params_stubs = xname
    
        params_data = lzip([self.forg(params[i],4) for i in exog_idx],
                       [self.forg(std_err[i]) for i in exog_idx],
                       [self.forg(tvalues[i]) for i in exog_idx],
                       [self.forg(pvalues[i]) for i in exog_idx],
                       [self.forg(conf_int[i,0]) for i in exog_idx],
                       [self.forg(conf_int[i,1]) for i in exog_idx],
                       [self.forg(vif[i]) for i in exog_idx])
        parameter_table = SimpleTable(params_data,
                                      param_header,
                                      params_stubs,
                                  title=title
                                  )

        return parameter_table
    
    def cov_params(self, r_matrix=None, column=None, scale=None, cov_p=None,
                   other=None):
        """
        Compute the variance/covariance matrix.

        The variance/covariance matrix can be of a linear contrast of the
        estimated parameters or all params multiplied by scale which will
        usually be an estimate of sigma^2.  Scale is assumed to be a scalar.

        Parameters
        ----------
        r_matrix : array_like
            Can be 1d, or 2d.  Can be used alone or with other.
        column : array_like, optional
            Must be used on its own.  Can be 0d or 1d see below.
        scale : float, optional
            Can be specified or not.  Default is None, which means that
            the scale argument is taken from the model.
        cov_p : ndarray, optional
            The covariance of the parameters. If not provided, this value is
            read from `self.normalized_cov_params` or
            `self.cov_params_default`.
        other : array_like, optional
            Can be used when r_matrix is specified.

        Returns
        -------
        ndarray
            The covariance matrix of the parameter estimates or of linear
            combination of parameter estimates. See Notes.
        """
        dot_fun = np.dot
        if r_matrix is not None:
            r_matrix = np.asarray(r_matrix)
            if r_matrix.shape == ():
                raise ValueError("r_matrix should be 1d or 2d")
            if other is None:
                other = r_matrix
            else:
                other = np.asarray(other)
            tmp = dot_fun(r_matrix, dot_fun(cov_p, np.transpose(other)))
            return tmp
        else:  # if r_matrix is None and column is None:
            return cov_p
    def wald_test(self, r_matrix, xname=None,cov_p=None, scale=1.0, invcov=None,
                  use_f=None):
        
        """
        Compute a Wald-test for a joint linear hypothesis.

        Parameters
        ----------
        r_matrix : {array_like, str, tuple}
            One of:

            - array : An r x k array where r is the number of restrictions to
              test and k is the number of regressors. It is assumed that the
              linear combination is equal to zero.
            - str : The full hypotheses to test can be given as a string.
              See the examples.
            - tuple : A tuple of arrays in the form (R, q), ``q`` can be
              either a scalar or a length p row vector.

        cov_p : array_like, optional
            An alternative estimate for the parameter covariance matrix.
            If None is given, self.normalized_cov_params is used.
        scale : float, optional
            Default is 1.0 for no scaling.

            .. deprecated:: 0.10.0

        invcov : array_like, optional
            A q x q array to specify an inverse covariance matrix based on a
            restrictions matrix.
        use_f : bool
            If True, then the F-distribution is used. If False, then the
            asymptotic distribution, chisquare is used. If use_f is None, then
            the F distribution is used if the model specifies that use_t is True.
            The test statistic is proportionally adjusted for the distribution
            by the number of constraints in the hypothesis.
        df_constraints : int, optional
            The number of constraints. If not provided the number of
            constraints is determined from r_matrix.

        Returns
        -------
        ContrastResults
            The results for the test are attributes of this results instance.
        """
        from patsy import DesignInfo
        names = xname
        params = self.params.ravel()
        LC = DesignInfo(names).linear_constraint(r_matrix)
        r_matrix, q_matrix = LC.coefs, LC.constants

        cparams = np.dot(r_matrix, params[:, None])
        J = float(r_matrix.shape[0])  # number of restrictions

        if q_matrix is None:
            q_matrix = np.zeros(J)
        else:
            q_matrix = np.asarray(q_matrix)
        if q_matrix.ndim == 1:
            q_matrix = q_matrix[:, None]
            if q_matrix.shape[0] != J:
                raise ValueError("r_matrix and q_matrix must have the same "
                                 "number of rows")
        Rbq = cparams - q_matrix
        if invcov is None:
            cov_p = self.cov_params(r_matrix=r_matrix, cov_p=self.Hinv(self.params))
            if np.isnan(cov_p).max():
                raise ValueError("r_matrix performs f_test for using "
                                 "dimensions that are asymptotically "
                                 "non-normal")
            invcov = np.linalg.pinv(cov_p)
            J_ = np.linalg.matrix_rank(cov_p)
            if J_ < J:
                import warnings
                warnings.warn('covariance of constraints does not have full '
                              'rank. The number of constraints is %d, but '
                              'rank is %d' % (J, J_), ValueWarning)
                J = J_
            
        F = np.dot(np.dot(Rbq.T, invcov), Rbq)
        df_resid = self.df_resid
        
        return ContrastResults(chi2=F, df_denom=J, statistic=F,
                                   distribution='chi2', distargs=(J,))
        
        
        
        
    def wald_test_terms(self, xname=None,skip_single=False, extra_constraints=None,
                        combine_terms=None):
        """
        Compute a sequence of Wald tests for terms over multiple columns.

        This computes joined Wald tests for the hypothesis that all
        coefficients corresponding to a `term` are zero.
        `Terms` are defined by the underlying formula or by string matching.

        Parameters
        ----------
        skip_single : bool
            If true, then terms that consist only of a single column and,
            therefore, refers only to a single parameter is skipped.
            If false, then all terms are included.
        extra_constraints : ndarray
            Additional constraints to test. Note that this input has not been
            tested.
        combine_terms : {list[str], None}
            Each string in this list is matched to the name of the terms or
            the name of the exogenous variables. All columns whose name
            includes that string are combined in one joint test.

        Returns
        -------
        WaldTestResults
            The result instance contains `table` which is a pandas DataFrame
            with the test results: test statistic, degrees of freedom and
            pvalues.
        """
        # lazy import
        from collections import defaultdict

        if extra_constraints is None:
            extra_constraints = []
        if combine_terms is None:
            combine_terms = []

        identity = np.eye(len(self.params))
        constraints = []
        combined = defaultdict(list)
        if xname is None:
            xname = ['x_%d' % i for i in range(len(self.params))]
            xname[0] = 'const'
        else:
            xname = xname
        for col, name in enumerate(xname):
            constraint_matrix = np.atleast_2d(identity[col])
            constraints.append((name, constraint_matrix))
            
        combined_constraints = []
        for cname in combine_terms:
            combined_constraints.append((cname, np.vstack(combined[cname])))

        distribution = ['chi2']
        res_wald = []
        index = []
        for name, constraint in constraints + combined_constraints + extra_constraints:
            wt = self.wald_test(constraint,xname=xname)
            row = [wt.statistic.item(), wt.pvalue.item(), constraint.shape[0]]
            if self.use_t:
                row.append(wt.df_denom)
            res_wald.append(row)
            index.append(name)
        # distribution nerutral names
        col_names = ['wald_value', 'p_value', 'df_constraint']
        # TODO: maybe move DataFrame creation to results class
        from pandas import DataFrame
        table = DataFrame(res_wald, index=index, columns=col_names)
        # TODO: remove temp again, added for testing
        return table


class sklearn_logistic(logistic_output_table):
    """
    Example:
    --------------------------------------------------------------
    import numpy as np
    import pandas as pd
    from sklearn.datasets import load_boston
    from sklearn.linear_model import LogisticRegression
    X, y = load_boston(return_X_y=True)
    y_one = [1 if i >20 else 0 for i in y]
    y_one = pd.Series(y_one)
    import sklearn_logistic_model_weight as slmw
    sample_weight = [1.5 if i == 1 else 1 for i in y_one]
    sample_weight = np.array(sample_weight)
    sl = slmw.sklearn_logistic(X, y_one,sample_weight=sample_weight)
    res = sl.Logistic()
    --------------------------------------------------------------
    """
    
    def __init__(self,x_in,y_in,selection='stepwise',sle=0.05, sls=0.05,includes=[],sample_weight=None):
        self.x_in = x_in
        self.y_in = y_in
        self.selection = selection
        self.sle = sle
        self.sls = sls
        self.includes = includes
        self.sample_weight = sample_weight
        
    #计算卡方统计
    def score_test(self,x_data,y_data,y_pred,DF=1):
        x_arr = np.matrix(x_data)
        y_arr = np.matrix(y_data).reshape(-1,1)
        yh_arr= np.matrix(y_pred).reshape(-1,1)
        
        grad_0 = - x_arr.T * (y_arr - yh_arr)
        info_0 = np.multiply(x_arr, np.multiply(yh_arr, (1-yh_arr))).T * x_arr
        if np.linalg.det(info_0)==0.0:
            return (0,DF,1)
        elif np.linalg.det(info_0)!=0.0:
            cov_m = info_0**(-1)
            chi2_0 = grad_0.T * cov_m * grad_0
            Pvalue=(1-stats.chi2.cdf(chi2_0[0,0],DF))
            return (chi2_0[0,0],DF,Pvalue)
        
    def Logistic(self):

        ###检查x和y的长度
        if len(self.x_in) != len(self.y_in):
            raise paraException(mesg="x,y不一致！")
            
        x_data = self.x_in.copy()
        y_data = self.y_in.copy()
        if self.sample_weight is None:
            sample_weight = np.ones(len(y_data))
        else:
            sample_weight = self.sample_weight
        ###检查x
        if isinstance(x_data,pd.core.frame.DataFrame) == True:
            x_list = list(x_data.columns.copy())
        elif isinstance(x_data,np.ndarray) == True:
            if len(x_data.shape) == 1:
                x_list = ['x_0']
                x_data = pd.DataFrame(x_data.reshape(-1,1),columns=x_list)        
            elif len(x_data.shape) == 2:
                x_list = ['x_' + str(i) for i in np.arange(x_data.shape[1])]
                x_data = pd.DataFrame(x_data, columns=x_list)
            else:
                raise paraException(mesg="x有问题!")
        else:
            raise paraException(mesg="x有问题!")

        #print(x_list)
        
        ####处理强制进入变量
        try:
            if len(self.includes)>0:
                includes = x_list[:self.includes].copy()
            else:
                includes = []
        except:
            pass
        
        ####处理x，y
        x_data['const']=1
        
        if (isinstance(y_data,pd.core.frame.DataFrame) == True) or (isinstance(y_data,pd.core.series.Series) == True):
            y_data = y_data.values.reshape(-1,1)
        else:
            y_data = y_data.reshape(-1,1)
            
        ####Stepwise  
        if self.selection.upper() == "STEPWISE":
            include_list = ['const'] + includes   ##强制包含的变量
            current_list = []                     ##当前模型中包含的变量，除了include_list中的变量
            candidate_list = [_x for _x in x_list if _x not in include_list]         ##候选变量列表
            ##第一次拟合:
            lgt = LogisticRegression(penalty='none',solver='newton-cg',fit_intercept=False)
            res = lgt.fit(x_data[include_list], y_data,sample_weight = sample_weight)
            ##预测结果
            y_pred = res.predict_proba(x_data[include_list])[:,1]
            ####输出第一步的拟合结果以及卡方检验结果
            print('====================第 0 步结果====================')
            print(logistic_output_table(res,x_data[include_list]).\
                  summary(xname=list(x_data[include_list].columns)))
            print(logistic_output_table(res,x_data[include_list]).\
                  wald_test_terms(xname=list(x_data[include_list].columns)))
            
            ##循环增删变量
            STOP_FLAG = 0
            step_i = 1
            while(STOP_FLAG == 0):
            
                if len(candidate_list) == 0:
                    break
                
                ##遍历所有候选变量，计算每一个加入的候选变量对应的score统计量
                score_list = [self.score_test(x_data[include_list + current_list + [x0]]
                                        ,y_data
                                        ,y_pred)
                              for x0 in candidate_list]
                score_df=pd.DataFrame(score_list,columns=['chi2','df','p-value'])
                score_df['xvar']=candidate_list
                
                slt_idx=score_df['chi2'].idxmax()
                p_value = score_df['p-value'].iloc[slt_idx]
                enter_x = candidate_list[slt_idx]
                ###
                ####输出第i步的候选变量的score统计量
                print('######======第 ' + str(step_i) + ' 步：候选变量的遍历结果======')
                print(score_df[['xvar','chi2','df','p-value']])
                
                if p_value <= self.sle:
                    current_list.append(enter_x)     ##加入模型列表
                    candidate_list.remove(enter_x)   ##从候选变量列表中删除
                    print('######====第 ' + str(step_i) + ' 步：加入变量 ' + enter_x)
                else:
                    STOP_FLAG = 1
                    print('######====第 ' + str(step_i) + ' 步：未加入变量' )
                
                
                ##根据新的变量列表，重新拟合，查看是否所有变量都能保留
                lgt = LogisticRegression(penalty='none',solver='newton-cg',fit_intercept=False)
                res = lgt.fit(x_data[include_list + current_list], y_data,sample_weight = sample_weight)
                ##预测结果
                y_pred = res.predict_proba(x_data[include_list + current_list])[:,1]
                ##wald chi2 test
                try:
                    chi2_df = logistic_output_table(res,x_data[include_list + current_list]).\
                    wald_test_terms(xname = list(x_data[include_list + current_list].columns)).copy()
                    ##检查是否有变量需要被删除
                    tmp_del_list = [tmp_x for tmp_x in chi2_df.index if tmp_x not in include_list]
                
                    ####输出第i步的候选变量的score统计量
                    print('######======第 ' + str(step_i) + ' 步：wald卡方检验======')
                    print(chi2_df)
                    
                    ##如果p-value大于等于sls，删除最大的
                    if len(tmp_del_list) > 0:
                        tmp_chi2 = chi2_df.loc[tmp_del_list].sort_values(by='wald_value')
                        if tmp_chi2['p_value'].iloc[0] > self.sls:
                            del_x = tmp_chi2.index[0]
                            
                            ##打印结果
                            print('######====第 ' + str(step_i) + ' 步：删除变量 ' + del_x)
                            
                            ##如果删除的是最近加入的变量，则停止筛选
                            if del_x == current_list[-1]:
                                current_list.remove(del_x)
                                STOP_FLAG = 1
                            else:
                                current_list.remove(del_x)
                            
                            ##删除的变量加入候选变量列表中
                            candidate_list.append(del_x)
                            
                            ###根据删除后的变量列表再次拟合
                            lgt = LogisticRegression(penalty='none',solver='newton-cg',fit_intercept=False)
                            res = lgt.fit(x_data[include_list + current_list], y_data,sample_weight = sample_weight)
                            ##预测结果
                            y_pred = res.predict_proba(x_data[include_list + current_list])[:,1]
                    else:
                        print('######====第 ' + str(step_i) + ' 步：未删除变量' )
                except:
                    current_list.remove(enter_x)     ##将刚加进来的变量从现有变量集剔除
                    print('######====第 ' + str(step_i) + ' 步：加入的变量 ' + enter_x + ',造成Singular matrix，将剔除！')
                else:
                    print('######====第 ' + str(step_i) + ' 步：未删除变量' )
                      
                print('#########################################') 
                step_i += 1  
    
            print('######========================最终结果汇总========================')        
            print(logistic_output_table(res,x_data[include_list + current_list]).\
                  summary(xname=list(x_data[include_list + current_list].columns)))
            print(logistic_output_table(res,x_data[include_list + current_list]).\
                  wald_test_terms(xname=list(x_data[include_list + current_list].columns)))
            x_list = list(x_data[include_list + current_list].columns)
        
        ####简单逻辑回归 
        else:
            lgt = LogisticRegression(random_state=0,penalty='none',solver='newton-cg',fit_intercept=False)
            res = lgt.fit(x_data, y_data,sample_weight = sample_weight)
            x_list = list(x_data.columns)
            print('######========================最终结果汇总========================')
            print(logistic_output_table(res,x_data).summary(xname=list(x_data.columns)))
            print(logistic_output_table(res,x_data).wald_test_terms(xname=list(x_data.columns)))
        return res,x_list

class evaluate_model(sklearn_logistic):
    
    def __init__(self,estimator,X_te,y_te):
        self.estimator = estimator
        self.X_te = X_te
        self.y_te = y_te
   
    def cumulation(self,true_label, guess_label):
        """ 计算样本分布累积占比及对应的KS值 """
        cumulative_1 = plt.hist(guess_label[true_label == 1], bins=np.arange(0, 1, 0.001), color='blue', cumulative=1, histtype='step', label='Bad users')
        cumulative_2 = plt.hist(guess_label[true_label == 0], bins=np.arange(0, 1, 0.001), color='green', cumulative=1, histtype='step', label='Good users')
        return cumulative_1, cumulative_2, np.abs(cumulative_1[0] - cumulative_2[0])
    def evaluate(self,true_label, guess_label, hardCut=False):
        """
        模型性能统计分析
        Args:
            true_label: 测试样本真实标签序列
            guess_label: 测试样本预测标签序列
        returns:
            (aucv, precision, recall, accuracy, fscore, ks, actual_cut)
        """
        def logging(*params):
            print(datetime.now().strftime('%Y-%m-%d %H:%M:%S'), ' '.join(['%s' for _ in params]) % params) 
        true_label = column_or_1d(true_label)
        guess_label = column_or_1d(guess_label)
    
        cumulative_1, _, cumu_delta = self.cumulation(true_label, guess_label)
        ks = np.max(cumu_delta)
        softcut = cumulative_1[1][np.argmax(cumu_delta)]

        if isinstance(hardCut, float):
            actual_cut = hardCut
        else:
            hardCut = 0.5
            actual_cut = softcut
    
        fpr, tpr, _ = roc_curve(true_label, guess_label)
        A = sum(logical_and(guess_label >= actual_cut, true_label == 1))
        B = sum(logical_and(guess_label >= actual_cut, true_label == 0))
        C = sum(logical_and(guess_label < actual_cut, true_label == 1))
        D = sum(logical_and(guess_label < actual_cut, true_label == 0))
    
        accuracy = 1.0 * (A + D) / (A + B + C + D)
        precision = 1.0 * A / (A + B)
        acc_pos = 1.0 * A / (A + C)
        acc_neg = 1.0 * D / (B + D)
        recall = acc_pos
        gmean = sqrt(acc_pos * acc_neg)
        fscore = 2.0 * precision * recall / (precision + recall)
        aucv = auc(fpr, tpr)
        logging(u'实际类别为1的个数: %d, 判定类别为1的个数: %d' % (sum(true_label == 1), sum(guess_label >= actual_cut)))
        logging(u'实际类别为0的个数: %d, 判定类别为0的个数: %d' % (sum(true_label == 0), sum(guess_label < actual_cut)))
        logging(u'A=%d, B=%d, C=%d, D=%d' % (A, B, C, D))
        logging(u'Precision=%.4f, Recall=%.4f, Accuracy=%.4f' % (precision, recall, accuracy))
        logging(u'AUC:%.4f, G-mean=%.4f, F-score=%.4f' % (aucv, gmean, fscore))
        logging('KS=%.4f,' % ks, 'Softcut=%.4f,' % softcut, 'HardCut=%.4f' % hardCut)
        
        return (aucv, precision, recall, accuracy, fscore, ks, actual_cut)
    
    def visualization(self,true_label, guess_label):
        """
            可视化统计分析
        Args:
            true_label: 测试样本真实标签序列
            guess_label: 测试样本预测标签序列
        returns:
            None
        """

        plt.clf()
        plt.gcf().set_size_inches(22, 12)
    
        # 整体预判概率分布
        plt.subplot(2, 2, 1)
        plt.hist(guess_label, bins=50, color='green', weights=np.ones_like(guess_label) / len(guess_label))
        plt.grid()
        plt.xlabel(u'预测概率')
        plt.ylabel(u'用户占比')
        plt.title(u'整体预判概率分布')

        # ROC曲线
        fpr, tpr, _ = roc_curve(true_label, guess_label)
        plt.subplot(2, 2, 2)
        plt.plot(fpr, tpr, label='ROC Curve')
        plt.plot([0, 1], [0, 1], 'y--')
        plt.grid()
        plt.xlabel('False positive rate')
        plt.ylabel('True positive rate')
        plt.title(u'ROC曲线')
    
        # 正负类别概率分布
        plt.subplot(2, 2, 3)
        plt.hist(guess_label[true_label == 1], bins=50, color='blue',
                 weights=np.ones_like(guess_label[true_label == 1]) / len(guess_label[true_label == 1]), label='Bad users')
        plt.hist(guess_label[true_label == 0], bins=50, color='green', alpha=0.8,
                 weights=np.ones_like(guess_label[true_label == 0]) / len(guess_label[true_label == 0]), label='Good users')
        plt.grid()
        plt.xlabel(u'预测概率')
        plt.ylabel(u'用户占比')
        plt.title(u'正负类别概率分布')
        plt.legend(loc='best')

        # 概率累积分布
        plt.subplot(2, 2, 4)
        cumulative_1, _, cumu_delta = self.cumulation(true_label, guess_label)
        plt.plot(cumulative_1[1][1:], cumu_delta, color='red', label='KS Curve')
        plt.grid()
        plt.title(u'概率累积分布')
        plt.xlabel(u'预测概率')
        plt.ylabel(u'累积占比')
        plt.legend(loc='upper left')
        plt.show()
        
    def assess(self):
        '''
        模型指标评价
        :param estimator:
        :param X_te:
        :param Y_te:
        :param img_path:
        :return:
        '''
        # guess_label = estimator.predict_proba(X_te)[:, 1].reshape(-1)
        guess_labe = self.estimator.predict_proba(self.X_te)[:,1]
        guess_label = guess_labe.reshape(-1,1)
        guess_result=pd.DataFrame(guess_label)
        print(guess_label[:5])
        self.evaluate(self.y_te, guess_label)
        self.visualization(self.y_te, guess_label)
        
class logistic(evaluate_model):
    def __init__(self,x_train,y_train,x_test,y_test,selection='stepwise',sle=0.05, sls=0.05,includes=[],sample_weight=None):
        self.x_train = x_train
        self.y_train = y_train
        self.x_test = x_test
        self.y_test = y_test
        self.selection = selection
        self.sle = sle
        self.sls = sls
        self.includes = includes
        self.sample_weight = sample_weight
    
    def logistic_(self):
        def logging(*params):
            print(datetime.now().strftime('%Y-%m-%d %H:%M:%S'), ' '.join(['%s' for _ in params]) % params)
        
        logging(u"****************开始逐步回归*****************")
        estimator=sklearn_logistic(self.x_train, self.y_train,self.selection,self.sle,self.sls,self.includes,self.sample_weight)
        res,x_var = estimator.Logistic()
        logging(u"***********训练集性能评估*************")
        ###检查x
        if isinstance(self.x_train,pd.core.frame.DataFrame) == True:
            x_list = list(self.x_train.columns)
            x_train_data = self.x_train.copy()
        elif isinstance(self.x_train,np.ndarray) == True:
            if len(self.x_train.shape) == 1:
                x_list = ['x_0']
                x_train_data = pd.DataFrame(self.x_train.reshape(-1,1),columns=x_list)        
            elif len(self.x_train.shape) == 2:
                x_list = ['x_' + str(i) for i in np.arange(self.x_train.shape[1])]
                x_train_data = pd.DataFrame(self.x_train, columns=x_list)
            else:
                raise paraException(mesg="x有问题!")
        else:
            raise paraException(mesg="x有问题!")
        x_tr = sm.add_constant(x_train_data)
        x_tr = x_tr[x_var]
        evaluate_model(res, x_tr, self.y_train).assess()

        logging(u"***********测试集性能评估*************")
        ###检查x
        if isinstance(self.x_test,pd.core.frame.DataFrame) == True:
            x_list = list(self.x_test.columns)
            x_test_data = self.x_test.copy()
        elif isinstance(self.x_test,np.ndarray) == True:
            if len(self.x_test.shape) == 1:
                x_list = ['x_0']
                x_test_data = pd.DataFrame(self.x_test.reshape(-1,1),columns=x_list)        
            elif len(self.x_test.shape) == 2:
                x_list = ['x_' + str(i) for i in np.arange(self.x_test.shape[1])]
                x_test_data = pd.DataFrame(self.x_test, columns=x_list)
            else:
                raise paraException(mesg="x有问题!")
        else:
            raise paraException(mesg="x有问题!")
        x_te = sm.add_constant(x_test_data)
        x_te = x_te[x_var]
        evaluate_model(res,x_te, self.y_test).assess()
        return estimator

