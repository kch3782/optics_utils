from cmath import cos, sin, tan
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

    float32     = staticmethod(np.float32)
    float64     = staticmethod(np.float64)
    complex64   = staticmethod(np.complex64)
    complex128  = staticmethod(np.complex128)

    exp         = staticmethod(np.exp)
    sqrt        = staticmethod(np.sqrt)
    sin         = staticmethod(np.sin)
    cos         = staticmethod(np.cos)
    tan         = staticmethod(np.tan)
    arcsin      = staticmethod(np.arcsin)
    arccos      = staticmethod(np.arccos)
    arctan      = staticmethod(np.arctan)

    abs         = staticmethod(np.abs)
    real        = staticmethod(np.real)
    imag        = staticmethod(np.imag)
    conj        = staticmethod(np.conj)

    arange      = staticmethod(np.arange)
    meshgrid    = staticmethod(np.meshgrid)
    where       = staticmethod(np.where)

    dim         = staticmethod(np.ndim)
    expand      = staticmethod(np.expand_dims)
    squeeze     = staticmethod(np.squeeze)
    reshape     = staticmethod(np.reshape)

    matmul      = staticmethod(np.matmul)

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

        float32     = staticmethod(torch.float32)
        float64     = staticmethod(torch.float64)
        complex64   = staticmethod(torch.complex64)
        complex128  = staticmethod(torch.complex128)

        exp         = staticmethod(torch.exp)
        sqrt        = staticmethod(torch.sqrt)
        sin         = staticmethod(torch.sin)
        cos         = staticmethod(torch.cos)
        tan         = staticmethod(torch.tan)
        arcsin      = staticmethod(torch.arcsin)
        arccos      = staticmethod(torch.arccos)
        arctan      = staticmethod(torch.arctan)

        abs         = staticmethod(torch.abs)
        real        = staticmethod(torch.real)
        imag        = staticmethod(torch.imag)
        conj        = staticmethod(torch.conj)

        arange      = staticmethod(torch.arange)
        meshgrid    = staticmethod(torch.meshgrid)
        where       = staticmethod(torch.where)

        dim         = staticmethod(torch.Tensor.dim)
        expand      = staticmethod(torch.unsqueeze)
        squeeze     = staticmethod(torch.squeeze)
        reshape     = staticmethod(torch.reshape)

        matmul      = staticmethod(torch.matmul)

        fft2        = staticmethod(torch.fft.fft2)
        ifft2       = staticmethod(torch.fft.ifft2)
        fftfreq     = staticmethod(torch.fft.fftfreq)
        fftshift    = staticmethod(torch.fft.fftshift)
        ifftshift   = staticmethod(torch.fft.ifftshift)

functions = NumpyBackend()

def select_backend(name):
    # perform checks
    if name == 'torch' and not TORCH_AVAILABLE:
        raise ValueError("PyTorch backend is not available, PyTorch must be installed.")

    # change backend by monkeypatching
    if name == 'numpy':
        functions.__class__ = NumpyBackend
    elif name == 'torch':
        functions.__class__ = TorchBackend
    else:
        raise ValueError("unknown backend")