import numpy as np
from skimage.io import imread
from time import time


imatge_input=imread('/Users/mariamagdalenapolpujadas/Desktop/universitat/tfg/GITHUB/spline_registration/proves/dog_input.jpg')
imatge_reference= imread('/Users/mariamagdalenapolpujadas/Desktop/universitat/tfg/GITHUB/spline_registration/proves/dog_reference.jpg')

nx, ny = (20,20)
malla = np.mgrid[0:imatge_input.shape[0]:round(imatge_input.shape[0]/nx), 0:imatge_input.shape[1]:round(imatge_input.shape[1]/ny)]
malla_x = malla[0]#inicialitzam a on van les coordenades x a la imatge_reference
malla_y = malla[1]#inicialitzam a on van les coordenades y a la imatge_reference
coordenadesx = np.arange(0,imatge_input.shape[0],round(imatge_input.shape[0]/nx))
coordenadesy = np.arange(0,imatge_input.shape[1],round(imatge_input.shape[1]/ny))
delta = [coordenadesx[1],coordenadesy[1]]

A2x=malla_x.ravel()
A2y=malla_y.ravel()
A2=np.concatenate((A2x,A2y ), axis=0)
A = np.copy(A2)

def colors_transform(reference_image,A):

    '''
    #si li introduim A com un array_like with shape (n,) on n= nx*ny*2.
    #ara A es és una matriu de dimensió (2, nx*ny)
    # A[0] conté les coordenades x, A[1] conté les coordenades y.
    '''
    if A.shape == (2*nx*ny,):
        A = np.array([A[0:(nx * ny)], A[nx * ny:(2 * nx * ny)]])
        '''
        #si li introduim A com un array_like with shape (n,) on n= nx*ny*2.
        #ara A es és una matriu de dimensió (2, nx*ny)
        # A[0] conté les coordenades x, A[1] conté les coordenades y.
        '''
    A = np.maximum(A, 0)

    A[0] = np.minimum(A[0], reference_image.shape[0] - 2)
    A[1] = np.minimum(A[1], reference_image.shape[1] - 2)


    X = A[0]
    Y = A[1]
    Ux = np.floor(X).astype(int)
    Uxceil= Ux + 1

    Vy = np.floor(Y).astype(int)
    Vyceil = Vy + 1
    a = X-Ux
    b = Y-Vy

    #si a és 0 no hauriem de tenir en compte Ux+1 tan sols Ux i si b es 0 no he m de tenir en compte Vy+1 tan sols Vy
    M = np.array([reference_image[Ux, Vy][:, 0],reference_image[Ux, Vy][:, 1],reference_image[Ux, Vy][:, 2]])
    B = np.array([reference_image[Uxceil, Vy][:,0],reference_image[Uxceil, Vy][:,1],reference_image[Uxceil, Vy][:,2]])
    C = np.array([reference_image[Ux, Vyceil][:,0],reference_image[Ux, Vyceil][:,1],reference_image[Ux, Vyceil][:,2]])
    D = np.array([reference_image[Uxceil, Vyceil][:,0],reference_image[Uxceil, Vyceil][:,1],reference_image[Uxceil, Vyceil][:,2]])


    color = (a-1)*(b-1)*M + a*(1-b)*B + (1-a)*b*C + a*b*D

    return color

def posicio(x,y,malla_x,malla_y):

    s = x / delta[0] - np.floor(x / delta[0]) #val 0 quan la x està a coordenadesx
    t = y / delta[1] - np.floor(y / delta[1]) #val 0 quan la y està a coordenadesy
    i = np.floor(x / delta[0]).astype(int) #index de la posició més pròxima per davall de la coordenada x a la malla
    j = np.floor(y / delta[1]).astype(int) #index de la posició més pròxima per davall de la coordenada y a la malla

    A = np.array([malla_x[i, j], malla_y[i, j]])
    B = np.array([malla_x[i+1, j], malla_y[i+1, j]])
    C = np.array([malla_x[i, j+1], malla_y[i, j+1]])
    D = np.array([malla_x[i+1, j+1],malla_y[i+1,j+1]])
    interpolacio = (s-1)*(t-1)*A + s*(1-t)*B + (1-s)*t*C + s*t*D

    return interpolacio

def funcio_min(p):
    malla_x = p[0:nx*ny].reshape(nx,ny)
    malla_y = p[nx*ny:2*nx*ny].reshape(nx,ny)

    from spline_registration.utils import imatge_vec
    I = imatge_vec(imatge_input[0:coordenadesx[(nx - 1)], 0:coordenadesx[(ny - 1)]], 3)
    C = np.mgrid[0:coordenadesx[(nx - 1)], 0:coordenadesy[(ny - 1)]]
    Cx = C[0].ravel()
    Cy = C[1].ravel()
    P = posicio(Cx, Cy,malla_x,malla_y)
    R = colors_transform(imatge_reference, P)
    R = R.flatten('F')
    R = (R).reshape(coordenadesx[(nx - 1)] * coordenadesy[(ny - 1)], 3)

    dif = np.sum(np.abs(I-R), 1)
    return dif
from scipy.optimize import least_squares
topti=time()
resultat = least_squares(funcio_min, A2, method='lm')
tfi=time()
print(resultat, tfi-topti)
