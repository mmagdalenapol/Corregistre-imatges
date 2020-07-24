import numpy as np
from skimage.io import imread
from time import time



imatge_input=imread('/Users/mariamagdalenapolpujadas/Desktop/universitat/tfg/GITHUB/spline_registration/proves/dog_input.jpg')
imatge_reference= imread('/Users/mariamagdalenapolpujadas/Desktop/universitat/tfg/GITHUB/spline_registration/proves/dog_reference.jpg')

nx, ny = (10,10)
delta = [int(imatge_input.shape[0]/nx),int(imatge_input.shape[1]/ny)]
malla = np.mgrid[0:nx*delta[0]:delta[0], 0:ny*delta[1]:delta[1]]

malla_x = malla[0]#inicialitzam a on van les coordenades x a la imatge_reference
malla_y = malla[1]#inicialitzam a on van les coordenades y a la imatge_reference
coordenadesx = np.arange(0,nx*delta[0],delta[0])
coordenadesy = np.arange(0,ny*delta[1],delta[1])



malla_vector = np.concatenate((malla_x.ravel(),malla_y.ravel() ), axis=0)

def colors_transform_nearest_neighbours(reference_image, Coordenades):
    '''
    si li introduim Coordenades com un array_like with shape (n,) on n= nx*ny*2.
    ara Coordenades es és una matriu de dimensió (2, nx*ny)
        Coordenades[0] conté les coordenades x,
        Coordenades[1] conté les coordenades y.
    '''
    Coordenades = Coordenades.astype('int')     # Discretitzar
    Coordenades = np.maximum(Coordenades, 0)
    Coordenades[0] = np.minimum(Coordenades[0], reference_image.shape[0] - 2)
    Coordenades[1] = np.minimum(Coordenades[1], reference_image.shape[1] - 2)
    registered_image = reference_image[Coordenades[0], Coordenades[1]]

    registered_image = registered_image.reshape(reference_image.shape,order='F')
    return registered_image


def posicio(x, y, malla_x, malla_y):
    # s = x / delta[0] - np.floor(x / delta[0])   # val 0 quan la x està a coordenadesx
    # t = y / delta[1] - np.floor(y / delta[1])   # val 0 quan la y està a coordenadesy
    # i = (x / delta[0]).astype(int)      # index de la posició més pròxima per davall de la coordenada x a la malla
    # j = (y / delta[1]).astype(int)      # index de la posició més pròxima per davall de la coordenada y a la malla

    s, i = np.modf(x/delta[0])
    t, j = np.modf(y/delta[0])

    i = np.minimum(np.maximum(i.astype('int'), 0), nx-2)
    j = np.minimum(np.maximum(j.astype('int'), 0), ny-2)

    # A = np.array([malla_x[i, j], malla_y[i, j]])
    # B = np.array([malla_x[i+1, j], malla_y[i+1, j]])
    # C = np.array([malla_x[i, j+1], malla_y[i, j+1]])
    # D = np.array([malla_x[i+1, j+1], malla_y[i+1,j+1]])
    interpolacio = np.array([
        (s-1)*(t-1)*malla_x[i, j] + s*(1-t)*malla_x[i+1, j] + (1-s)*t*malla_x[i, j+1] + s*t*malla_x[i+1, j+1],
        (s-1)*(t-1)*malla_y[i, j] + s*(1-t)*malla_y[i+1, j] + (1-s)*t*malla_y[i, j+1] + s*t*malla_y[i+1, j+1],
        ])

    return interpolacio

def funcio_min(p):
    titer = time()
    malla_x = p[0:(nx)*(ny)].reshape(nx, ny)
    malla_y = p[(nx)*(ny):2*(nx)*(ny)].reshape(nx, ny)



    x = np.arange(imatge_input.shape[0])
    y = np.arange(imatge_input.shape[1])
    Cx, Cy = np.meshgrid(x, y)
    Cx = Cx.ravel()
    Cy = Cy.ravel()
    P = posicio(Cx, Cy,malla_x,malla_y)
    R = colors_transform_nearest_neighbours(imatge_reference, P)
    #R = colors_transform_bilinear_interpolation(imatge_reference, P)

    # R = R.flatten('F')
    # R = (R).reshape(coordenadesx[(nx - 1)] * coordenadesy[(ny - 1)], 3)
    # dif = np.sum(np.abs(I-R), 1)

    I=imatge_input

    dif = np.sum(np.power(imatge_input - R, 2))

    tfi = time()
    # print('temps',tfi-titer )

    return dif
from scipy.optimize import least_squares
topti=time()
resultat = least_squares(funcio_min, malla_vector, method='trf',verbose=2)
tfi=time()
print(resultat, tfi-topti)
parametres_optims = resultat.x

malla_x = parametres_optims[0:(nx) * (ny)].reshape(nx, ny )
malla_y = parametres_optims[(nx) * (ny):2 * (nx) * (ny)].reshape(nx, ny)

x = np.arange(imatge_input.shape[0])
y = np.arange(imatge_input.shape[1])
Cx, Cy = np.meshgrid(x, y)
Cx = Cx.ravel()
Cy = Cy.ravel()
P = posicio(Cx, Cy, malla_x, malla_y)
R = colors_transform_nearest_neighbours(imatge_reference, P)







from matplotlib import pyplot as plt
import pylab as pl
#VEURE MALLA SOBRE LA IMATGE IMPUT

plt.imshow(imatge_input)
pl.plot(malla[0],malla[1],color='blue')
pl.plot(malla[1],malla[0],color='blue')
pl.title('malla inicial sobre la imatge input')
pl.show()

plt.imshow(R)
pl.plot(malla_x,malla_y,color='green')
pl.plot(malla_y,malla_x,color = 'green')
pl.title('malla òptima sobre la imatge registrada')
pl.show()
