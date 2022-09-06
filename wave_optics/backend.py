import numpy as np

try:
    import torch
    import torch.fft
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False

class NumpyBackend():
    is_tensor   = False
    pi          = staticmethod(np.pi)
    array       = staticmethod(np.array)

    exp         = staticmethod(np.exp)
    sqrt        = staticmethod(np.sqrt)
    real        = staticmethod(np.real)
    imag        = staticmethod(np.imag)
    conj        = staticmethod(np.conj)

    arange      = staticmethod(np.arange)
    meshgrid    = staticmethod(np.meshgrid)
    where       = staticmethod(np.where)
    expand      = staticmethod(np.expand_dims)

    fft2        = staticmethod(np.fft.fft2)
    ifft2       = staticmethod(np.fft.ifft2)
    fftfreq     = staticmethod(np.fft.fftfreq)
    fftshift    = staticmethod(np.fft.fftshift)
    ifftshift   = staticmethod(np.fft.ifftshift)

if TORCH_AVAILABLE:
    class TorchBackend():
        is_tensor   = True
        pi          = staticmethod(np.pi)
        array       = staticmethod(torch.tensor)

        exp         = staticmethod(torch.exp)
        sqrt        = staticmethod(torch.sqrt)
        real        = staticmethod(torch.real)
        imag        = staticmethod(torch.imag)
        conj        = staticmethod(torch.conj)

        arange      = staticmethod(torch.arange)
        meshgrid    = staticmethod(torch.meshgrid)
        where       = staticmethod(torch.where)
        expand      = staticmethod(torch.unsqueeze)

        fft2        = staticmethod(torch.fft.fft2)
        ifft2       = staticmethod(torch.fft.ifft2)
        fftfreq     = staticmethod(torch.fft.fftfreq)
        fftshift    = staticmethod(torch.fft.fftshift)
        ifftshift   = staticmethod(torch.fft.ifftshift)

backend = NumpyBackend()

def select_backend(name):
    # perform checks
    if name == 'torch' and not TORCH_AVAILABLE:
        raise ValueError("PyTorch backend is not available, PyTorch must be installed.")

    # change backend by monkeypatching
    if name == 'numpy':
        backend.__class__ = NumpyBackend
    elif name == 'torch':
        backend.__class__ = TorchBackend
    else:
        raise ValueError("unknown backend")