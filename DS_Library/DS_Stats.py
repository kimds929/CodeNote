import numpy as np
from collections import namedtuple
import scipy as sp

def anova(*args, equal_var=False):
    # https://svn.r-project.org/R/trunk/src/library/stats/R/oneway.test.R
    # translated from R Welch ANOVA (not assuming equal variance)
    if equal_var:
        return sp.stats.f_oneway(*[list(arg) for arg in args])
    else:
        F_onewayResult = namedtuple('F_onewayResult', ('statistic', 'pvalue'))
        args = [np.asarray(list(arg), dtype=float) for arg in args]
        k = len(args)
        ni =np.array([len(arg) for arg in args])
        mi =np.array([np.mean(arg) for arg in args])
        vi =np.array([np.var(arg,ddof=1) for arg in args])
        wi = ni/vi

        tmp =sum((1-wi/sum(wi))**2 / (ni-1))
        tmp /= (k**2 -1)

        dfbn = k - 1
        dfwn = 1 / (3 * tmp)

        m = sum(mi*wi) / sum(wi)
        f = sum(wi * (mi - m)**2) /((dfbn) * (1 + 2 * (dfbn - 1) * tmp))
        prob = sp.special.fdtrc(dfbn, dfwn, f)   # equivalent to stats.f.sf
        return F_onewayResult(f, prob)
