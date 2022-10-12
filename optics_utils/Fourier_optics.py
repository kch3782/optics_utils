from .backend import functions as fn
from . import Fourier_utils

# Angular Spectrum Method
class ASM:
    # Transfer function channels [batch, lamb0, dz, x=N[0], y=N[1]]
    def __init__(self, N: list[int], d: list[float], dz: list[float], lamb0: list[float], sign = 1, dtype = 'float', device = None):

        self.device = device
        self.dtype = fn.float64 if dtype == 'double' else fn.float32

        self.N = fn.array(N)
        self.d = fn.array(d,dtype=self.dtype)
        self.dz = self._expand_channels(fn.array(dz,dtype=self.dtype),dims=[-2,-2,-1,-1])
        self.lamb0 = self._expand_channels(fn.array(lamb0,dtype=self.dtype),dims=[-2,-1,-1,-1])
        self.sign = int(sign)
        
        self.fx = fn.fftshift(fn.fftfreq(N[0],d=d[0]))
        self.fy = fn.fftshift(fn.fftfreq(N[1],d=d[1]))

        self._set_device()

        Fx, Fy = fn.meshgrid(self.fx,self.fy,indexing='ij')

        # Fx = self._expand_channels(Fx,dims=[-3,-3,-3])
        # Fy = self._expand_channels(Fx,dims=[-3,-3,-3])

        self.mask = Fx**2 + Fy**2 < (self.lamb0)**(-2)

        self.Hpow = self.sign*1.j*2*fn.pi*self.dz*fn.sqrt((self.lamb0)**(-2)-Fx**2-Fy**2+0.j)
        self.H = fn.exp(fn.where(fn.real(self.Hpow)<0.,self.Hpow,-fn.conj(self.Hpow)))
    
    def _expand_channels(self,x,dims:list[int]):
        for dim in dims:
            x = fn.expand(x,dim)
        return x

    def _set_device(self):
        if fn.is_tensor and (self.device is not None):
            self.N = self.N.to(self.device)
            self.d = self.d.to(self.device)
            self.dz = self.dz.to(self.device)
            self.lamb0 = self.lamb0.to(self.device)

            self.fx = self.fx.to(self.device)
            self.fy = self.fy.to(self.device)

    def propagate(self,X,mask='on'):
        Xf = fn.fftshift(fn.fft2(fn.ifftshift( X )))

        if mask == 'off':
            Y = fn.fftshift(fn.ifft2(fn.ifftshift( self.H*Xf )))
        else:
            Y = fn.fftshift(fn.ifft2(fn.ifftshift( self.H*self.mask*Xf )))

        return Y

class RotationalSymmetricASM:
        # Transfer function channels [batch=1, lamb0, dz, r=N]
    def __init__(self, N: int, r_max: int, dz: list[float], lamb0: list[float], sign = 1, dtype = 'float', device = None):

        self.device = device
        self.dtype = fn.float64 if dtype == 'double' else fn.float32

        self.N = fn.array(N)
        self.r_max = fn.array(r_max,dtype=self.dtype)
        self.dz = self._expand_channels(fn.array(dz,dtype=self.dtype),dims=[0,0,-1])
        self.lamb0 = self._expand_channels(fn.array(lamb0,dtype=self.dtype),dims=[0,-1,-1])
        self.sign = int(sign)

        self.QDHT = Fourier_utils.QDHT(N, r_max, dtype = self.dtype, device = device)

        self._set_device()

        self.r = self.QDHT.r
        self.fr = self.QDHT.fr

        self.mask = self.fr < (self.lamb0)**(-1)

        self.Hpow = self.sign*1.j*2*fn.pi*self.dz*fn.sqrt((self.lamb0)**(-2)-self.fr**2+0.j)
        self.H = fn.exp(fn.where(fn.real(self.Hpow)<0.,self.Hpow,-fn.conj(self.Hpow)))
    
    def _expand_channels(self,x,dims:list[int]):
        for dim in dims:
            x = fn.expand(x,dim)
        return x

    def _set_device(self):
        if fn.is_tensor and (self.device is not None):
            self.N = self.N.to(self.device)
            self.r_max = self.r_max.to(self.device)
            self.dz = self.dz.to(self.device)
            self.lamb0 = self.lamb0.to(self.device)

    def propagate(self,X,mask='on'):
        Xfr = self.QDHT.forward(X)

        if mask == 'off':
            Y = self.QDHT.inverse( self.H*Xfr )
        else:
            Y = self.QDHT.inverse( self.H*self.mask*Xfr )

        return Y
