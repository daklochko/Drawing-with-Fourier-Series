import numpy as np
import matplotlib.pyplot as plt
from matplotlib import animation
from math import pi, cos, sin
from scipy.integrate import quad
from PIL import Image

image_name = 'note.png'
image = Image.open(image_name).convert('1')
contour = plt.contour(image, origin = 'image')
path = contour.collections[0].get_paths()[0]
X, Y = path.vertices[:,0], path.vertices[:,1]
X = X - min(X)
Y = Y - min(Y)
X = X - max(X)/2
Y = Y - max(Y)/2

def f(t, tp, xp, yp):
    X = np.interp(t, tp, xp) 
    Y = 1j*np.interp(t, tp, yp)
    return X + Y

def Cn(tp, xp, yp, N):
    Cn = []
    for n in range(-N, N + 1):
        real = quad(lambda t: np.real(f(t, tp, xp, yp)*np.exp(-n*1j*t)), 0, 2*pi)[0]/(2*pi)
        imag = quad(lambda t: np.imag(f(t, tp, xp, yp)*np.exp(-n*1j*t)), 0, 2*pi)[0]/(2*pi)
        Cn.append([real, imag])
    return np.array(Cn)

def Fourier_series(t, Cn, N):
    exp = np.array([np.exp(-n*1j*t) for n in range(-N, N + 1)])
    series = np.sum((Cn[:,0] + 1j*Cn[:,1]) * exp[:])
    return np.real(series), np.imag(series)

