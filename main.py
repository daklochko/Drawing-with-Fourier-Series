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
xmin, xmax = plt.xlim()
ymin, ymax = plt.ylim()

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


def visualize(x_Fourier, y_Fourier, coeffs, N, space, fig_lim):

    fig, ax = plt.subplots(figsize = (10, 10))
    lim = max(fig_lim)
    ax.set_xlim([-lim, lim])
    ax.set_ylim([-lim, lim])
    ax.set_aspect('equal')
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)

    line = plt.plot([], [], 'k', linewidth = 5)[0]
    radius = [plt.plot([], [], 'r', linewidth = 1)[0] for i in range(2*N + 1)]
    circles = [plt.plot([], [], 'r', linewidth = 1)[0] for i in range(2*N + 1)]

    def update_c(c, t):
        c_new = []
        for i, j in enumerate(range(-N, N + 1)):
            dtheta = -j*t
            v = [cos(dtheta)*c[i][0] - sin(dtheta)*c[i][1], cos(dtheta)*c[i][1] + sin(dtheta)*c[i][0]]
            c_new.append(v)
        return np.array(c_new)

    def sort_velocity(N):
        idx = []
        for i in range(1, N + 1):
            idx.extend([N + i, N - i]) 
        return idx    

    def animate(i):
        line.set_data(x_Fourier[:i], y_Fourier[:i])
        r = [np.linalg.norm(coeffs[j]) for j in range(len(coeffs))]
        pos = coeffs[N]
        c = update_c(coeffs, i/len(space) * 2*pi)
        idx = sort_velocity(N)
        for j, rad, circle in zip(idx, radius, circles):
            new_pos = pos + c[j]
            rad.set_data([pos[0], new_pos[0]], [pos[1], new_pos[1]])
            theta = np.linspace(0, 2*pi, 50)
            x, y = r[j]*np.cos(theta) + pos[0], r[j]*np.sin(theta) + pos[1]
            circle.set_data(x, y)
            pos = new_pos
            
    ani = animation.FuncAnimation(fig, animate, frames = len(space), interval = 10)
    return ani

N = 30
tt = np.linspace(0, 2*pi, len(X))
coeffs = Cn(tt, X, Y, N)
space = np.linspace(0, 2*pi, 300)
x_Fourier = [Fourier_series(t, coeffs, N)[0] for t in space]
y_Fourier = [Fourier_series(t, coeffs, N)[1] for t in space]

anim = visualize(x_Fourier, y_Fourier, coeffs, N, space, [xmin, xmax, ymin, ymax])
anim.save('note.gif', writer = 'pillow')

