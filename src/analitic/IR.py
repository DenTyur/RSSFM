import math

import numpy as np


class IR(object):
    """
    Ionization Rate
    """

    def __init__(self, I, Z=1, l=0, m=0, A=1):
        self.I = I
        self.Z = Z
        self.l = l
        self.m = m
        self.A = A

        self.alpha = 0

        self.E0 = None
        self.E = None

    def set_E0(self, E0):
        self.E0 = E0

    def set_Et(self, E):
        self.E = E

        # ========== PPT ===========================

    def Ckl(self):
        v = ((self.Z**2) / (2 * self.I)) ** (1 / 2)
        k = (2 * self.I) ** (1 / 2)  # k = Z/v
        return self.A / (2 * k ** (v + 1 / 2))

    def B(self):
        numerator = (2 * self.l + 1) * math.factorial(self.l + np.abs(self.m))
        denominator = (
            2 ** (2 * np.abs(self.m))
            * math.factorial(np.abs(self.m))
            * math.factorial(self.l - np.abs(self.m))
        )
        return numerator / denominator

    def nu(self):
        return self.Z / np.sqrt(2 * self.I)

    def F0(self):
        return self.E0 / (2 * self.I) ** (3 / 2)

    def F(self, t):
        return self.E(t) / (2 * self.I) ** (3 / 2)
        
    def Ecr(self):
        return (2 * self.I) ** (3 / 2)

    def get_w_PPT(self, E0):
        self.set_E0(E0)
        w = (
            2 ** (2 * self.nu() + 1)
            * self.Ckl() ** 2
            * self.B()
            * self.I
            * self.F0() ** (1 + self.m - 2 * self.nu())
            * np.exp(-2 / (3 * self.F0()))
        )
        return w

    def get_w_PPT_only_exp(self, E0):
        self.set_E0(E0)
        w = (
            self.F0() ** (1 + self.m - 2 * self.nu())
            * np.exp(-2 / (3 * self.F0()))
        )
        return w

    def get_wt_PPT(self, t):
        w = (
            2 ** (2 * self.nu() + 1)
            * self.Ckl() ** 2
            * self.B()
            * self.I
            * self.F(t) ** (1 + self.m - 2 * self.nu())
            * np.exp(-2 / (3 * self.F(t)))
        )
        return w
        
        
        

    # ============= ADK ==========================

    def get_Eb(self):
        k = np.sqrt(2 * self.I)
        return k**4 / (16 * self.Z)

    def get_w_ADK(self, E0, C_ADK=1):
        k = np.sqrt(2 * self.I)
        mf = math.factorial(np.abs(self.m))
        w1 = C_ADK**2 / (2 ** np.abs(self.m) * mf)
        w2 = (
            (2 * self.l + 1)
            * math.factorial(self.l + np.abs(self.m))
            / (2 * math.factorial(self.l - np.abs(self.m)))
        )
        w3 = 1 / (k ** (2 * self.Z / k - 1))
        w4 = (2 * k**3 / E0) ** (2 * self.Z / k - np.abs(self.m) - 1)
        w5 = np.exp(-2 * k**3 / (3 * E0))
        return w1 * w2 * w3 * w4 * w5

    # ================ TongLin ===========================
    def set_alpha(self, alpha):
        self.alpha = alpha
        
    def get_wt_TongLin(self, t):
        corr = np.exp(-self.alpha*
                      (self.Z**2/self.I)*
                      (self.E(t)/(2*self.I)**(3/2))
                     )
        return self.get_wt_PPT(t) * corr