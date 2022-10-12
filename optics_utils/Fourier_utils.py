import scipy.special
from .backend import functions as fn

# Quasi Discrete Hankel Transform
class QDHT:
    # Channels [... , r=N]
    def __init__(self, N: int, r_max: float, dtype = 'float', device = None):
        
        self.device = device
        self.dtype = fn.float64 if dtype == 'double' else fn.float32

        self.N = fn.array(N)
        self.r_max = fn.array(r_max,dtype=self.dtype)

        J0_zeros = fn.array(scipy.special.jn_zeros(0,N+1),dtype=self.dtype)
        self.J0_zeros = J0_zeros[:-1]
        S = J0_zeros[-1]

        self.J1_J0_zeros = scipy.special.j1(self.J0_zeros)
        self.Cmn = 2/S * scipy.special.j0(fn.reshape(self.J0_zeros,[-1,1])*fn.reshape(self.J0_zeros,[1,-1])/S) *\
            (fn.abs(fn.reshape(self.J1_J0_zeros,[-1,1]))**(-1) *
            fn.abs(fn.reshape(self.J1_J0_zeros,[1,-1]))**(-1))

        self.fr_max = S/(2*fn.pi*self.r_max)

        self.r = self.J0_zeros/(2*fn.pi*self.fr_max)
        self.fr = self.J0_zeros/(2*fn.pi*self.r_max)

        self._set_device()

    def _set_device(self):
        if fn.is_tensor and (self.device is not None):
            self.N = self.N.to(self.device)

            self.r = self.r.to(self.device)
            self.fr = self.fr.to(self.device)

            self.r_max = self.r_max.to(self.device)
            self.fr_max = self.fr_max.to(self.device)

            self.J0_zeros = self.J0_zeros.to(self.device)
            self.J1_J0_zeros = self.J1_J0_zeros.to(self.device)
            self.Cmn = self.Cmn.to(self.device)

    def forward(self,f):
        f_mod = f * self.r_max / fn.abs(self.J1_J0_zeros)

        if fn.is_tensor:
            F_mod = fn.squeeze(fn.matmul(self.Cmn.to(f),fn.expand(f_mod,-1)),-1)
        else:
            F_mod = fn.squeeze(fn.matmul(self.Cmn,fn.expand(f_mod,-1)),-1)
        
        F = F_mod / self.fr_max * fn.abs(self.J1_J0_zeros)

        return F
    
    def inverse(self,F):
        F_mod = F * self.fr_max / fn.abs(self.J1_J0_zeros)
        if fn.is_tensor:
            f_mod = fn.squeeze(fn.matmul(self.Cmn.to(F),fn.expand(F_mod,-1)),-1)
        else:
            f_mod = fn.squeeze(fn.matmul(self.Cmn,fn.expand(F_mod,-1)),-1)
        
        f = f_mod / self.r_max * fn.abs(self.J1_J0_zeros)

        return f
