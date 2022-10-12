from .backend import functions as fn

def J2S(Ex,Ey):
    # Jones vector to Stokes parameter
    S0 = fn.abs(Ex)**2 + fn.abs(Ey)**2
    S1 = fn.abs(Ex)**2 - fn.abs(Ey)**2
    S2 = 2*fn.real(Ex*fn.conj(Ey))
    S3 = - 2*fn.imag(Ex*fn.conj(Ey))
    
    return S0, S1, S2, S3

def S2J(S1,S2,S3):
    # Stokes vector to Jones vector
    norm = fn.sqrt(S1**2 + S2**2 + S3**2)
    S1 /= norm
    S2 /= norm
    S3 /= norm

    Ax = fn.sqrt((1+S1)/2)
    Ay = fn.sqrt((1-S1)/2)

    if fn.abs(S1) == 1.:
        phi_y = 0.
    else:
        if (S2 >= 0.):
            phi_y = fn.arcsin(S3/fn.sqrt(1.-S1**2))
        elif (S3 >= 0.):
            phi_y = fn.arccos(S2/fn.sqrt(1.-S1**2))
        else:
            phi_y = -fn.arccos(S2/fn.sqrt(1.-S1**2))

    Ex = Ax*fn.exp(1.j*0.)
    Ey = Ay*fn.exp(1.j*phi_y)

    return Ex, Ey