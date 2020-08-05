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
    #titer = time()
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

    #tfi = time()
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

## PROBLEMES PIXELS NEGRES IMATGE CORREGISTRADA

parametres_optims =np.array([-10.99581752, -14.47496419,  -5.64779476,  -5.83839859,
         0.69268295, -18.43661499, -12.27038855,  80.7149934 ,
        59.36776442,  69.52567756,  70.09744048,  67.70195709,
        62.32214752,  59.0526673 , 168.61695113, 161.77788156,
       168.16368659, 163.83623566, 161.73277285, 159.83271614,
       149.06937466, 253.22012207, 273.43694728, 249.1171973 ,
       250.43639677, 251.00049794, 238.61093544, 236.81203612,
       343.59029898, 345.56696664, 339.47761292, 342.30802322,
       341.46329652, 337.9332355 , 327.52135038, 430.46911463,
       429.19092789, 425.2874871 , 409.46799823, 418.172036  ,
       399.5720362 , 408.26504823, 523.83440367, 523.50942826,
       517.13342051, 510.9406087 , 509.13649425, 504.2042596 ,
       507.63142705,  -6.16820491,  91.74399646, 158.58706986,
       222.95408579, 304.20434201, 384.80084422, 490.60235922,
       -15.4908522 ,  93.75347933, 155.46820886, 211.61785356,
       335.03156928, 421.32243996, 499.81185347,   2.79799969,
       104.6497987 , 131.44920035, 226.2629881 , 329.55437555,
       421.91426341, 504.14830571, -17.43657589,  95.60224234,
       151.78706634, 239.35858436, 315.69737251, 415.16368893,
       505.95164754,  -3.71687896,  64.76500956, 155.79140628,
       242.10469046, 327.32734939, 400.16327502, 504.18181092,
        17.57528615,  95.66031799, 164.80329435, 251.77140223,
       329.00774826, 443.26168965, 510.97618829,  11.88295369,
        84.59607614, 165.70851739, 256.67142074, 337.23156415,
       428.91105384, 516.93123354])

malla_x = parametres_optims[0:(nx+1)*(ny+1)].reshape((nx+1),(ny+1))
malla_y = parametres_optims[(nx+1)*(ny+1):2*(nx+1)*(ny+1)].reshape((nx+1),(ny+1))

x = np.arange(imatge_input.shape[0])
y = np.arange(imatge_input.shape[1])
Coord_originals_x, Coord_originals_y  = np.meshgrid(x, y)
Coord_originals_x = Coord_originals_x.ravel()
Coord_originals_y  = Coord_originals_y .ravel()
Coordenades_desti = posicio(Coord_originals_x, Coord_originals_y, malla_x, malla_y)
Coordenades_desti = np.maximum(Coordenades_desti,0 )
Coordenades_desti [0] = np.minimum(Coordenades_desti [0], imatge_input.shape[0] - 1)
Coordenades_desti [1] = np.minimum(Coordenades_desti [1], imatge_input.shape[1] - 1)
posicions=Coordenades_desti.reshape(2,imatge_input.shape[0], imatge_input.shape[1])

R=imatge_reference[np.round(posicions[1]).astype(int), np.round(posicions[0]).astype(int)]

#PROVES OPCIO1
parametres_optims=np.array([-1.56223681e+00, -2.22508810e-01, -4.05181505e+00, -4.85517091e+00,
       -9.18613687e+00, -1.78916731e+01, -1.06515915e+01,  8.03490321e+01,
        7.84815831e+01,  7.86619662e+01,  7.53843228e+01,  7.07691860e+01,
        6.76111755e+01,  6.43322554e+01,  1.65544482e+02,  1.66625246e+02,
        1.61417073e+02,  1.65358244e+02,  1.61354383e+02,  1.55321653e+02,
        1.50510969e+02,  2.54849992e+02,  2.61821730e+02,  2.49783460e+02,
        2.55755523e+02,  2.46712491e+02,  2.40633718e+02,  2.39756095e+02,
        3.44874002e+02,  3.44114585e+02,  3.37912156e+02,  3.36114402e+02,
        3.33607362e+02,  3.25160555e+02,  3.30489698e+02,  4.30166130e+02,
        4.29418450e+02,  4.24939865e+02,  4.17002635e+02,  4.18179279e+02,
        4.07479073e+02,  4.18414671e+02,  5.17662259e+02,  5.19911060e+02,
        5.15214045e+02,  5.10668133e+02,  5.11218195e+02,  5.06758172e+02,
        5.10991165e+02, -2.45847251e+00,  7.89466214e+01,  1.55764711e+02,
        2.35467450e+02,  3.17548915e+02,  4.12014990e+02,  5.07681263e+02,
       -6.14373131e+00,  8.00834020e+01,  1.53295592e+02,  2.27772965e+02,
        3.12810324e+02,  3.96927031e+02,  4.95626975e+02,  7.25253980e+00,
        9.12192351e+01,  1.59825591e+02,  2.41149720e+02,  3.21292486e+02,
        4.02827877e+02,  4.97436093e+02, -9.65895216e+00,  7.05121473e+01,
        1.52416843e+02,  2.40722242e+02,  3.22082358e+02,  4.06625205e+02,
        5.02266252e+02,  2.87237599e-01,  6.85877958e+01,  1.54747674e+02,
        2.44114277e+02,  3.23239345e+02,  4.11625950e+02,  5.08455627e+02,
       -2.35420186e+00,  8.56528873e+01,  1.66892292e+02,  2.47383059e+02,
        3.27807990e+02,  4.21085626e+02,  5.08893895e+02,  4.32086269e+00,
        8.51736121e+01,  1.69003506e+02,  2.54840663e+02,  3.39098481e+02,
        4.25579425e+02,  5.12557317e+02])

