import numpy as np
from skimage.io import imread
from time import time


imatge_input=imread('/Users/mariamagdalenapolpujadas/Desktop/universitat/tfg/GITHUB/spline_registration/proves/dog_input.jpg')
imatge_reference= imread('/Users/mariamagdalenapolpujadas/Desktop/universitat/tfg/GITHUB/spline_registration/proves/dog_reference.jpg')

nx, ny = (6,6)
delta = [int(imatge_input.shape[0]/nx),int(imatge_input.shape[1]/ny)]
malla = np.mgrid[0:nx*delta[0]:delta[0], 0:ny*delta[1]:delta[1]]

malla_x = malla[0]#inicialitzam a on van les coordenades x a la imatge_reference
malla_y = malla[1]#inicialitzam a on van les coordenades y a la imatge_reference
coordenadesx = np.arange(0,nx*delta[0],delta[0])
coordenadesy = np.arange(0,ny*delta[1],delta[1])



malla_vector = np.concatenate((malla_x.ravel(),malla_y.ravel() ), axis=0)

mallav_original = np.copy(malla_vector)

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

def imatge_transformada(imatge, coord_desti):

    '''
    Introduim la imatge_input i les coordenades a les quals es mouen les originals després d'aplicar l'interpolació.
    El que volem es tornar la imatge registrada que tengui a les coordenades indicades els colors originals:

    Per fer-ho definesc una imatge registrada (inicialment tota negre) i a les coordenades del destí
    anar enviant els colors originals.
    '''

    coord_desti = np.round(coord_desti).astype('int')     # Discretitzar
    coord_desti = np.maximum(coord_desti, 0)
    coord_desti[0] = np.minimum(coord_desti[0], imatge.shape[0] - 1)
    coord_desti[1] = np.minimum(coord_desti[1], imatge.shape[1] - 1)


    x = np.arange(coordenadesx[nx - 1]-1)
    y = np.arange(coordenadesy[ny - 1]-1)
    Coord_originals_x, Coord_originals_y = np.meshgrid(x, y)
    Coord_originals_x = Coord_originals_x.ravel()
    Coord_originals_y = Coord_originals_y.ravel()

    registered_image = np.zeros_like(imatge)
    registered_image[coord_desti[0], coord_desti[1]] = imatge[Coord_originals_x, Coord_originals_y]
    return registered_image

def colors_transform_nearest_neighbours(imatge, Coordenades):
    '''
    si li introduim Coordenades com un array_like with shape (n,) on n= nx*ny*2.
    ara Coordenades es és una matriu de dimensió (2, nx*ny)
        Coordenades[0] conté les coordenades x,
        Coordenades[1] conté les coordenades y.

    if Coordenades.shape == (2*nx*ny,):
        Coordenades = np.array([Coordenades[0:(nx * ny)], Coordenades[nx * ny:(2 * nx * ny)]])

    '''

    Coordenades = np.round(Coordenades).astype('int')     # Discretitzar
    Coordenades = np.maximum(Coordenades, 0)
    Coordenades[0] = np.minimum(Coordenades[0], imatge.shape[0] - 1)
    Coordenades[1] = np.minimum(Coordenades[1], imatge.shape[1] - 1)
    registered_image = imatge[Coordenades[0], Coordenades[1]]
    registered_image = registered_image.reshape(imatge.shape,order='F')

    return registered_image


'''
aquesta forma es menys eficient

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

'''
def posicio(x, y, malla_x, malla_y):
    # s = x / delta[0] - np.floor(x / delta[0])   # val 0 quan la x està a coordenadesx
    # t = y / delta[1] - np.floor(y / delta[1])   # val 0 quan la y està a coordenadesy
    # i = (x / delta[0]).astype(int)      # index de la posició més pròxima per davall de la coordenada x a la malla
    # j = (y / delta[1]).astype(int)      # index de la posició més pròxima per davall de la coordenada y a la malla

    s, i = np.modf(x/delta[0])
    t, j = np.modf(y/delta[0])


    i = np.minimum(np.maximum(i.astype('int'), 0), nx-1)
    j = np.minimum(np.maximum(j.astype('int'), 0), ny-1)

    # A = np.array([malla_x[i, j], malla_y[i, j]])
    # B = np.array([malla_x[i+1, j], malla_y[i+1, j]])
    # C = np.array([malla_x[i, j+1], malla_y[i, j+1]])
    # D = np.array([malla_x[i+1, j+1], malla_y[i+1,j+1]])
    interpolacio = np.array([
        (s-1)*(t-1)*malla_x[i, j] + s*(1-t)*malla_x[i+1, j] + (1-s)*t*malla_x[i, j+1] + s*t*malla_x[i+1, j+1],
        (s-1)*(t-1)*malla_y[i, j] + s*(1-t)*malla_y[i+1, j] + (1-s)*t*malla_y[i, j+1] + s*t*malla_y[i+1, j+1],
        ])

    return interpolacio

def funcio_min(parametres):

    malla_x = parametres[0:nx * ny].reshape(nx, ny)
    malla_y = parametres[nx * ny:2 * nx * ny].reshape(nx, ny)

    x = np.arange(coordenadesx[nx - 1]-1)
    y = np.arange(coordenadesy[ny - 1]-1)
    Coord_originals_x, Coord_originals_y = np.meshgrid(x, y)
    Coord_originals_x = Coord_originals_x.ravel()
    Coord_originals_y = Coord_originals_y.ravel()

    Coordenades_desti = posicio(Coord_originals_x, Coord_originals_y, malla_x, malla_y)
    imatge_registrada = imatge_transformada(imatge_input, Coordenades_desti)

    dif = np.sum(np.power(imatge_registrada[0:coordenadesx[nx-2],0:coordenadesy[ny-2]]-imatge_reference[0:coordenadesx[nx-2],0:coordenadesy[ny-2]] , 2))
    #dif = np.sum(np.abs(imatge_registrada[0:(coordenadesx[nx - 1]-1),0:(coordenadesy[ny - 1]-1)]-imatge_reference[0:(coordenadesx[nx - 1]-1),0:(coordenadesy[ny - 1]-1)] ), 2) #UN ERROR PER CADA PIXEL
    return dif
    #return dif.flatten()


def min_info_mutua(parametres):
    malla_x = parametres[0:nx * ny].reshape(nx, ny)
    malla_y = parametres[nx * ny:2 * nx * ny].reshape(nx, ny)

    x = np.arange(coordenadesx[nx - 1]-1)
    y = np.arange(coordenadesy[ny - 1]-1)
    Coord_originals_x, Coord_originals_y = np.meshgrid(x, y)
    Coord_originals_x = Coord_originals_x.ravel()
    Coord_originals_y = Coord_originals_y.ravel()

    Coordenades_desti = posicio(Coord_originals_x, Coord_originals_y, malla_x, malla_y)
    imatge_registrada = imatge_transformada(imatge_input, Coordenades_desti)
    from spline_registration.losses import info_mutua
    info = info_mutua(imatge_registrada[0:(coordenadesx[nx - 1]-1),0:(coordenadesy[ny - 1]-1)],imatge_reference[0:(coordenadesx[nx - 1]-1),0:(coordenadesy[ny - 1]-1)],5)
    return -info

from scipy.optimize import least_squares
#topti=time()
#resultatlm = least_squares(funcio_min, malla_vector, method='lm')
#tfi=time()
#print(resultatlm, tfi-topti)
'''
malla_optima_lm= np.array([ 6.07796566e-01,  1.46710292e+02,  2.66995869e+01,  2.13154961e+01,
        6.37798545e+00, -4.80483022e+01,  7.95887197e+01,  1.21487639e+02,
       -3.84195074e+00,  9.59346477e+01,  3.38093541e+01,  6.57791579e+01,
        9.65956217e+01,  3.79650520e+01,  2.12192935e+02,  1.68799898e+02,
        1.87079350e+02,  2.32097413e+02,  2.29552638e+02,  1.94307089e+02,
        1.55860769e+02,  2.55379878e+02,  2.13590784e+02,  2.11178690e+02,
        3.15386141e+02,  3.76526241e+02,  2.81595626e+02,  2.48404063e+02,
        3.22500755e+02,  3.47488651e+02,  3.79185554e+02,  3.24472699e+02,
        3.22095286e+02,  3.57020567e+02,  3.45022203e+02,  3.53259419e+02,
        4.66683656e+02,  4.98643191e+02,  3.16368409e+03,  4.55960241e+02,
        4.28737374e+02,  3.57886009e+02,  4.94105510e+02,  5.32729881e+02,
        4.35554801e+02,  4.93871369e+02,  4.27838055e+02,  4.17702322e+02,
        5.23592309e+02, -8.63298771e+00,  1.21733253e+02,  1.91556533e+02,
        2.97983210e+02,  4.08537463e+02,  4.42068633e+02,  5.15121775e+02,
       -3.38635495e+01,  9.15003267e+01,  1.73510926e+02,  1.88888547e+02,
        4.30801241e+02,  4.91132767e+02,  5.14603188e+02, -3.83282499e-01,
        1.02069970e+02,  1.83102194e+02,  1.92191297e+02,  3.93472631e+02,
        4.29235567e+02,  5.18848977e+02, -5.73221846e+01,  1.61941935e+02,
        1.70000000e+02,  2.55000000e+02,  3.40000000e+02,  4.25000000e+02,
        5.10000000e+02,  0.00000000e+00,  8.50000000e+01,  1.70000000e+02,
        2.55000000e+02,  3.40000000e+02,  4.25000000e+02,  5.10000000e+02,
        0.00000000e+00,  8.50000000e+01,  1.70000000e+02,  2.55000000e+02,
        3.40000000e+02,  4.25000000e+02,  5.10000000e+02,  0.00000000e+00,
        8.50000000e+01,  1.70000000e+02,  2.55000000e+02,  3.40000000e+02,
        4.25000000e+02,  5.10000000e+02])

temps_emprat =  4661.987810134888 #segons
'''

'''
  active_mask: array([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
       0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
       0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
       0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
       0, 0, 0, 0, 0, 0, 0, 0, 0, 0])
        cost: 2117327941.060306
         fun: array([39.1766103 , 32.        , 28.72698592, ..., 45.75473588,
       36.79805895, 42.02228388])
        grad: array([-9.78554477e+02,  2.21622265e+02,  4.45563729e+02,  2.07862698e+02,
        1.93595322e+01, -1.44758484e+00, -2.26694878e+03,  1.25504520e+03,
       -1.43808233e+03,  1.18677528e+04, -4.22632901e+03,  2.02087435e+04,
        1.03220124e+03, -1.19114885e+04, -1.75306442e+03,  1.88692630e+02,
        6.25198052e+03, -2.18027624e+03, -3.02026731e+03, -1.41068579e+04,
       -7.89861100e+03, -2.45836562e+03, -3.94804969e+02,  3.84436893e+03,
        7.96817800e+03, -4.31157441e+03, -3.83439507e+02,  4.23526189e+03,
       -2.70394229e+03, -1.51467519e+03, -1.20905806e+03,  1.08830962e+03,
        3.85990559e+03,  3.67792636e+02, -2.36875943e+03, -1.17819051e+03,
       -6.42227306e+02, -6.39858030e+02,  1.49761390e+01, -1.66398144e+02,
        7.43557989e+00,  2.09043349e+02, -1.50431717e+03, -4.22630094e+03,
        4.16222286e+03,  5.27766504e+03, -1.08677128e+01,  6.13299600e+03,
       -2.32945540e+03, -1.62005016e+04,  6.52009960e+03,  1.09367299e+04,
       -3.18854767e+03, -6.38051529e+03, -7.38138946e+02, -5.05083073e+03,
       -1.28668855e+03,  1.40659721e+03, -5.44107549e+03, -1.18809962e+03,
        9.39604589e+01,  2.82964965e+03, -1.53939787e+03, -2.63429505e+03,
       -8.16259434e+03, -1.64018160e+02,  2.69002392e+03,  4.88738370e+03,
        8.65631172e+02,  4.63783583e+03, -2.51643923e+03,  1.62769527e+03,
        0.00000000e+00,  0.00000000e+00,  0.00000000e+00,  0.00000000e+00,
        0.00000000e+00,  0.00000000e+00,  0.00000000e+00,  0.00000000e+00,
        0.00000000e+00,  0.00000000e+00,  0.00000000e+00,  0.00000000e+00,
        0.00000000e+00,  0.00000000e+00,  0.00000000e+00,  0.00000000e+00,
        0.00000000e+00,  0.00000000e+00,  0.00000000e+00,  0.00000000e+00,
        0.00000000e+00,  0.00000000e+00,  0.00000000e+00,  0.00000000e+00,
        0.00000000e+00,  0.00000000e+00])
         jac: array([[-3.00000000e+00,  0.00000000e+00,  0.00000000e+00, ...,
         0.00000000e+00,  0.00000000e+00,  0.00000000e+00],
       [ 2.34678583e-09,  0.00000000e+00,  0.00000000e+00, ...,
         0.00000000e+00,  0.00000000e+00,  0.00000000e+00],
       [-5.85882353e+00, -1.41176471e-01,  0.00000000e+00, ...,
         0.00000000e+00,  0.00000000e+00,  0.00000000e+00],
       ...,
       [ 0.00000000e+00,  0.00000000e+00,  0.00000000e+00, ...,
         0.00000000e+00,  0.00000000e+00,  0.00000000e+00],
       [ 0.00000000e+00,  0.00000000e+00,  0.00000000e+00, ...,
         0.00000000e+00,  0.00000000e+00,  0.00000000e+00],
       [ 0.00000000e+00,  0.00000000e+00,  0.00000000e+00, ...,
         0.00000000e+00,  0.00000000e+00,  0.00000000e+00]])
     message: '`ftol` termination condition is satisfied.'
        nfev: 30404
        njev: None
  optimality: 20208.74351646013
      status: 2
     success: True
           x: array([ 6.07796566e-01,  1.46710292e+02,  2.66995869e+01,  2.13154961e+01,
        6.37798545e+00, -4.80483022e+01,  7.95887197e+01,  1.21487639e+02,
       -3.84195074e+00,  9.59346477e+01,  3.38093541e+01,  6.57791579e+01,
        9.65956217e+01,  3.79650520e+01,  2.12192935e+02,  1.68799898e+02,
        1.87079350e+02,  2.32097413e+02,  2.29552638e+02,  1.94307089e+02,
        1.55860769e+02,  2.55379878e+02,  2.13590784e+02,  2.11178690e+02,
        3.15386141e+02,  3.76526241e+02,  2.81595626e+02,  2.48404063e+02,
        3.22500755e+02,  3.47488651e+02,  3.79185554e+02,  3.24472699e+02,
        3.22095286e+02,  3.57020567e+02,  3.45022203e+02,  3.53259419e+02,
        4.66683656e+02,  4.98643191e+02,  3.16368409e+03,  4.55960241e+02,
        4.28737374e+02,  3.57886009e+02,  4.94105510e+02,  5.32729881e+02,
        4.35554801e+02,  4.93871369e+02,  4.27838055e+02,  4.17702322e+02,
        5.23592309e+02, -8.63298771e+00,  1.21733253e+02,  1.91556533e+02,
        2.97983210e+02,  4.08537463e+02,  4.42068633e+02,  5.15121775e+02,
       -3.38635495e+01,  9.15003267e+01,  1.73510926e+02,  1.88888547e+02,
        4.30801241e+02,  4.91132767e+02,  5.14603188e+02, -3.83282499e-01,
        1.02069970e+02,  1.83102194e+02,  1.92191297e+02,  3.93472631e+02,
        4.29235567e+02,  5.18848977e+02, -5.73221846e+01,  1.61941935e+02,
        1.70000000e+02,  2.55000000e+02,  3.40000000e+02,  4.25000000e+02,
        5.10000000e+02,  0.00000000e+00,  8.50000000e+01,  1.70000000e+02,
        2.55000000e+02,  3.40000000e+02,  4.25000000e+02,  5.10000000e+02,
        0.00000000e+00,  8.50000000e+01,  1.70000000e+02,  2.55000000e+02,
        3.40000000e+02,  4.25000000e+02,  5.10000000e+02,  0.00000000e+00,
        8.50000000e+01,  1.70000000e+02,  2.55000000e+02,  3.40000000e+02,
        4.25000000e+02,  5.10000000e+02]) 4661.987810134888
'''

topti=time()
resultattrf = least_squares(funcio_min, malla_vector, method='trf')
tfi=time()
print(resultattrf, tfi-topti)

'''

malla_optima_trf=np.array([ 1.23366244e+01,  1.50829486e+02,  5.68818702e+01,  2.82007908e+01,
        2.17768029e+00, -5.17816159e+01,  7.88082203e+01,  1.31359969e+02,
       -2.04530153e+00,  9.78211318e+01,  4.99642898e+01,  6.79225152e+01,
        9.48386066e+01,  2.87750914e+01,  2.20382507e+02,  1.64715812e+02,
        1.76181233e+02,  2.22028095e+02,  2.30264406e+02,  1.99321251e+02,
        1.53227621e+02,  2.55052219e+02,  2.42191441e+02,  2.24889767e+02,
        3.14096381e+02,  3.74635882e+02,  2.82779210e+02,  2.48615694e+02,
        2.56698231e+02,  3.74109571e+02,  3.79395374e+02,  3.26169129e+02,
        3.18372968e+02,  3.64733900e+02,  3.49134913e+02,  3.47446036e+02,
        4.65358251e+02,  4.50993899e+02,  4.75248733e+02,  4.80580533e+02,
        4.21150925e+02,  3.68547385e+02,  4.74103884e+02,  5.28450157e+02,
        4.78906071e+02,  4.76601811e+02,  4.31873749e+02,  3.91382012e+02,
        5.30795548e+02, -1.39055523e+01,  7.48349575e+01,  2.11892700e+02,
        2.91646579e+02,  4.38425832e+02,  4.32173113e+02,  5.18116407e+02,
       -4.16278699e+01,  9.30385243e+01,  1.80243972e+02,  1.70781953e+02,
        4.40240633e+02,  4.89943355e+02,  5.17785514e+02, -4.61419339e-01,
        1.33143200e+02,  1.69801344e+02,  1.91955692e+02,  3.83486097e+02,
        4.26988718e+02,  5.26334092e+02, -7.55757686e+01,  1.73757221e+02,
       -5.83539815e+20, -6.37530686e+20, -5.58565113e+20,  7.57214161e+19,
       -1.31179465e+20, -5.30993186e+20, -6.44954422e+20, -8.29483643e+19,
       -1.03730575e+21,  6.86966343e+20,  5.22776235e+20,  4.09214472e+20,
       -9.49186243e+20,  6.86833044e+19,  3.02607033e+20, -4.09641520e+20,
        4.35817215e+20, -2.15101453e+20, -1.59515510e+20, -7.13640679e+20,
        3.39303993e+20,  8.07204465e+20, -1.07261710e+20, -5.14093914e+20,
       -1.23253218e+21, -9.64466280e+20])

temps_trf= 1722.3859059810638
'''

#topti=time()
#resultat = least_squares(funcio_min, malla_vector, method='trf', verbose=2, max_nfev=10)
#tfi=time()
#print(resultat, tfi-topti)

#topti=time()
#resultatinfo = least_squares(min_info_mutua, malla_vector, method='trf', verbose=2)#, max_nfev=10)
#tfi=time()
#print(resultatinfo, tfi-topti)







parametres_optims = resultattrf.x

malla_x = parametres_optims[0:(nx) * (ny)].reshape(nx, ny )
malla_y = parametres_optims[(nx) * (ny):2 * (nx) * (ny)].reshape(nx, ny)

x = np.arange(coordenadesx[nx - 1]-1)
y = np.arange(coordenadesy[ny - 1]-1)
Coord_originals_x, Coord_originals_y  = np.meshgrid(x, y)
Coord_originals_x = Coord_originals_x.ravel()
Coord_originals_y  = Coord_originals_y .ravel()
Coordenades_desti = posicio(Coord_originals_x, Coord_originals_y, malla_x, malla_y)

imatge_registrada = imatge_transformada(imatge_input, Coordenades_desti)






from spline_registration.utils import visualize_side_by_side
from matplotlib import pyplot as plt

visualize_side_by_side(imatge_registrada,imatge_reference,'registrada i reference')

#gif amb les 2 imatges la registrada i la de referència
import imageio
images=[imatge_registrada, imatge_reference]
imageio.mimwrite('movie.gif', images, fps=1)

from PIL import Image

im1 = Image.open('/Users/mariamagdalenapolpujadas/Desktop/universitat/tfg/GITHUB/spline_registration/proves/dog_input.jpg').convert('LA')
im2=Image.open('/Users/mariamagdalenapolpujadas/Desktop/universitat/tfg/GITHUB/spline_registration/proves/dog_reference.jpg').convert('LA')

im = Image.blend(im1, im2, 0.5)
plt.imshow(im)
plt.title('imatge input sobre la imatge_reference')
plt.show()

import pylab as pl
#VEURE MALLA SOBRE LA IMATGE IMPUT


plt.imshow(imatge_input)
pl.plot(malla[0],malla[1],color='blue')
pl.plot(malla[1],malla[0],color='blue')
pl.title('malla inicial sobre la imatge input')
pl.show()


#VEURE A ON VA A PARAR LA MALLA I AMB QUINS VALORS HO COMPARA

plt.imshow(imatge_registrada)
pl.plot(malla_x,malla_y,color='green')
pl.plot(malla_y,malla_x,color = 'green')
pl.title('malla òptima sobre la imatge registrada')
pl.show()

#veim que no funciona perquè no canvia la malla i per tant la registrada es igual a la input (als punts que hi ha interpolació)
plt.imshow(imatge_registrada-imatge_input)
plt.show()


'''

from scipy.ndimage import gaussian_filter
im_gaussian = gaussian_filter(imatge_reference, sigma=5)


def min_gaussian_info_mutua(parametres):
    malla_x = parametres[0:nx * ny].reshape(nx, ny)
    malla_y = parametres[nx * ny:2 * nx * ny].reshape(nx, ny)

    x = np.arange(coordenadesx[nx - 1]-1)
    y = np.arange(coordenadesy[ny - 1]-1)
    Coord_originals_x, Coord_originals_y = np.meshgrid(x, y)
    Coord_originals_x = Coord_originals_x.ravel()
    Coord_originals_y = Coord_originals_y.ravel()

    Coordenades_desti = posicio(Coord_originals_x, Coord_originals_y, malla_x, malla_y)
    imatge_registrada = imatge_transformada(imatge_input, Coordenades_desti)
    from spline_registration.losses import info_mutua
    info = -info_mutua(imatge_registrada,im_gaussian,5)
    return info

res = least_squares(min_gaussian_info_mutua,malla_vector, method='trf',verbose=2)
parametres_optims = res.x

malla_x = parametres_optims[0:(nx) * (ny)].reshape(nx, ny )
malla_y = parametres_optims[(nx) * (ny):2 * (nx) * (ny)].reshape(nx, ny)

x = np.arange(coordenadesx[nx - 1]-1)
y = np.arange(coordenadesy[ny - 1]-1)
Coord_originals_x, Coord_originals_y  = np.meshgrid(x, y)
Coord_originals_x = Coord_originals_x.ravel()
Coord_originals_y  = Coord_originals_y .ravel()
Coordenades_desti = posicio(Coord_originals_x, Coord_originals_y, malla_x, malla_y)

imatge_registrada = imatge_transformada(imatge_input, Coordenades_desti)


from spline_registration.utils import visualize_side_by_side
from matplotlib import pyplot as plt

visualize_side_by_side(imatge_registrada,imatge_reference,'registrada i reference')
'''