from . import backend as bd

class AngularSpecgtrumMethod:
    # Transfer function channels [batch=1, lamb0, dz, x=N[0], y=N[1]]
    def __init__(self, N: list[int], d: list[float], dz: list[float], lamb0: list[float], sign = 1, device = None):

        self.N = bd.array(N)
        self.d = bd.array(d)
        self.dz = self._expand_channels(bd.array(dz),dims=[0,0,-1,-1])
        self.lamb0 = self._expand_channels(bd.array(lamb0),dims=[0,-1,-1,-1])
        self.sign = int(sign)
        self.device = device

        self.fx = bd.fftshift(bd.fftfreq(N[0],d=d[0]))
        self.fy = bd.fftshift(bd.fftfreq(N[1],d=d[1]))

        self._set_device()

        Fx, Fy = bd.meshgrid(self.fx,self.fy,indexing='ij')

        self.mask = Fx**2 + Fy**2 < (self.lamb0)**(-2)

        self.Hpow = self.sign*1.j*2*bd.pi*self.dz*bd.sqrt((self.lamb0)**(-2)-Fx**2-Fy**2+0.j)
        self.H = bd.exp(bd.where(bd.real(self.Hpow)<0.,self.Hpow,-bd.conj(self.Hpow)))
    
    def _expand_channels(self,x,dims:list[int]):
        for dim in dims:
            x = bd.expand(x,dim)
        return x

    def _set_device(self):
        if bd.is_tensor and (self.device is not None):
            self.N = self.N.to(self.device)
            self.d = self.d.to(self.device)
            self.dz = self.dz.to(self.device)
            self.lamb0 = self.lamb0.to(self.device)

            self.fx = self.fx.to(self.device)
            self.fy = self.fy.to(self.device)

    def propagate(self,X,mask='on'):
        Xf = bd.fftshift(bd.fft2(bd.ifftshift( X )))

        if mask == 'off':
            Y = bd.fftshift(bd.ifft2(bd.ifftshift( self.H*Xf )))
        else:
            Y = bd.fftshift(bd.ifft2(bd.ifftshift( self.H*self.mask*Xf )))

        return Y