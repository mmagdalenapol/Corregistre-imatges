import numpy as np
from skimage.io import imread, imsave
from time import time
from matplotlib import pyplot
import matplotlib.pyplot as plt
from skimage.filters import gaussian

imatge_input=imread('/Users/mariamagdalenapolpujadas/Desktop/universitat/tfg/GITHUB/spline_registration/proves/dog_input.jpg')
imatge_reference= imread('/Users/mariamagdalenapolpujadas/Desktop/universitat/tfg/GITHUB/spline_registration/proves/dog_reference.jpg')

sigma= 50
imatge_input_gaussian = gaussian(imatge_input, sigma=sigma, multichannel=True)
imatge_reference_gaussian = gaussian(imatge_reference,sigma=sigma,multichannel=True )

nx, ny = (6,6)
delta = [int(imatge_input.shape[0]/nx)+1,int(imatge_input.shape[1]/ny)+1]
'''
el +1 ens permet assegurar que la darrera fila/columna de la malla estan defora de la imatge.
Ja que així creant aquests punts ficticis a fora podem interpolar totes les posicions de la imatge. 
Ara la malla serà (nx+1)*(ny+1) però la darrera fila i la darrera columna com he dit són per tècniques.
'''
malla = np.mgrid[ 0:(nx+1)*delta[0]:delta[0], 0:(ny+1)*delta[1]:delta[1] ]

malla_x = malla[0]#inicialitzam a on van les coordenades x a la imatge_reference
malla_y = malla[1]#inicialitzam a on van les coordenades y a la imatge_reference


coordenadesx = np.arange(0,(nx+1)*delta[0],delta[0])
coordenadesy = np.arange(0,(ny+1)*delta[1],delta[1])



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

    x = np.arange(imatge.shape[0])
    y = np.arange(imatge.shape[1])

    Coord_originals_x, Coord_originals_y = np.meshgrid(x, y)
    Coord_originals_x = Coord_originals_x.ravel()
    Coord_originals_y = Coord_originals_y.ravel()

    registered_image = np.zeros_like(imatge)
    #registered_image = np.copy(imatge)
    #registered_image[coord_desti[0], coord_desti[1]] = imatge[Coord_originals_x, Coord_originals_y]
    registered_image[Coord_originals_x, Coord_originals_y] = imatge[coord_desti[0], coord_desti[1]]
    return registered_image

def colors_transform_nearest_neighbours(imatge, Coordenades):
    '''
        Coordenades[0] conté les coordenades x,
        Coordenades[1] conté les coordenades y.
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
    t, j = np.modf(y/delta[1])


    #i = np.minimum(np.maximum(i.astype('int'), 0), nx-1)
    #j = np.minimum(np.maximum(j.astype('int'), 0), ny-1)

    i = np.minimum(np.maximum(i.astype('int'), 0), nx)
    j = np.minimum(np.maximum(j.astype('int'), 0), ny)
    interpolacio = np.array([
        (s-1)*(t-1)*malla_x[i, j] + s*(1-t)*malla_x[i+1, j] + (1-s)*t*malla_x[i, j+1] + s*t*malla_x[i+1, j+1],
        (s-1)*(t-1)*malla_y[i, j] + s*(1-t)*malla_y[i+1, j] + (1-s)*t*malla_y[i, j+1] + s*t*malla_y[i+1, j+1]
        ])

    return interpolacio

#1a opcio
def min_imatge_transformada(parametres):

    malla_x = parametres[0:(nx+1)*(ny +1)].reshape((nx+1),(ny +1))
    malla_y = parametres[(nx+1)*(ny+1): 2*(nx+1)*(ny+1)].reshape((nx+1),(ny +1))

    x = np.arange(imatge_input.shape[0])
    y = np.arange(imatge_input.shape[1])

    Coord_originals_x, Coord_originals_y = np.meshgrid(x, y)
    Coord_originals_x = Coord_originals_x.ravel()
    Coord_originals_y = Coord_originals_y.ravel()

    Coordenades_desti = posicio(Coord_originals_x, Coord_originals_y, malla_x, malla_y)
    imatge_registrada = imatge_transformada(imatge_input, Coordenades_desti)


    #dif = np.sum(np.power(imatge_registrada-imatge_reference, 2))
    #dif = np.sum(np.abs(imatge_registrada-imatge_reference), 2) #UN ERROR PER CADA PIXEL

    '''
    dif = np.power(imatge_registrada-imatge_reference, 2)
    dif = np.sum(dif,2)
    dif = np.sqrt (dif)
    N = (imatge_input.shape[0])*(imatge_input.shape[1])
    dif = np.sum(dif)/N

    '''



    #cas filtratge gausià:

    dif2 = np.power(imatge_registrada-imatge_reference_gaussian, 2)
    dif2 = np.sum(dif2,2)
    dif2 = np.sqrt (dif2)
    N = (imatge_input.shape[0])*(imatge_input.shape[1])
    dif2 = np.sum(dif2)/N



    #regularitzar, tenint en compte que la distància entre els punts consecutis (tant de dreta a esquerra, com d'adalt abaix) siguin pròxims.
    #per fer això, el que vull és que la distància entre els punts en mitjana no sigui ni molt major ni molt menor que la inicial.



    '''
    si tenim els següents punts de la malla:
    
    A=(x_{n0,0},y_{n0,0})    B=(x_{n0,1},y_{n1,0})
    *                           *
    
    
    
    
    C=(x_{n1,0},y_{n0,1})     D=(x_{n1,1},y_{n1,})
    *                          *
    
    m' interessa que la distància distància de A a B i la de A a C sigui ni molt gran ni molt petita.
    Per fer-ho he pensat que com inicialment la distància de A a B és delta[1] i la de A a C és delta[0], 
    llavors voldré que la mitjana de les distàncies sigui propera a n'aquestes inicials. De manera que punts que estaven
    enfora inicialment ara no estiguin molt aprop i que punts que estaven aprop ara no estiguin molt enfora.
    
    Però clar no hem de mirar només la mitjana ja que se podrien compensar punts amb distàncies molt grans amb altres molt petites.
    Per tant el que vull és que a més es minimitzi també la desviació típica de les distàncies entre els punts de la malla final.
    
    així per dur-ho a terme tenint en compte que malla_x i malla_y tenen les següents estructura:
    
    malla_x= x_{n0,0} x_{n0,1} .... x_{n0,nx}           malla_y = y_{n0,0}  y_{n1,0} ... y_{ny,0}
             x_{n1,0} x_{n1,1} .... x_{n1,nx}                     y_{n0,1}  y_{n1,1} ... y_{ny,1}
                .                                                      .        .
                .                                                      .        .
                .                                                      .        .
             x_{nx,0} x_{nx,1} .... x_{nx,nx}                      y_{n0,ny} y_{n1,ny} ... y_{ny,ny}
    
    
    -començam per les distàncies d'esquerra a dreta (d1) aquí entés com la distància de A a B.
        d(A,B) = np.sqrt((x_{n0,1}-x_{n0,0})^2 + (y_{n1,0}-y_{n0,0})^2)
        
        així per a poder implementar-ho de manera relativament sencilla he pensat crear unes matrius que siguin 
        com malla_x i malla_y però que a les columnes 0 tenguin la 1 de les originals i així successivament. 
        Hem de tenir en compte que el que passi amb la darrera columna quedarà un poc enlaire i per això no ho tendrem en compte. 
        A n'aquestes matrius els anomen mx_col_post i my_col_post  respectivament.
        
        mx_col_post = x_{n0,1} x_{n0,2} .... x_{n0,nx} x_{n0,nx}            my_col_post= y_{n1,0}  y_{n2,0} ... y_{ny,0} y_{ny,0}
                      x_{n1,1} x_{n1,2} .... x_{n1,nx} x_{n1,nx}                         y_{n1,1}  y_{n2,1} ... y_{ny,1} y_{ny,1}
                         .                                                                   .        .
                         .                                                                   .        .
                         .                                                                   .        .
                      x_{nx,1} x_{nx,1} .... x_{nx,nx} x_{nx,nx}                          y_{n1,ny} y_{n2,ny} ... y_{ny,ny} y_{ny,ny}  
        
         
        per tant al cas general si consideram tots els punts de la malla i no tant sols dos d1 vendrà donat per:
        
        n= min(nx,ny)
        
        d1=np.sqrt(np.power((mx_col_post-malla_x),2)+np.power((my_col_post-malla_y),2))[0:n,0:n]
        
    - amb un raonament anàlog per les distàncies d'adalt a baix (d2) distància de A a C o de B a D.
        d(A,C) = np.sqrt((x_{n1,0}-x_{n0,0})^2 + (y_{n0,1}-y_{n0,0})^2)
        
        ara les matrius han tenir en compte a la fila 0 han de tenir el de la fila 1 i així successivament. A n'aquestes
        matrius els anomen mx_fila_post my_fila_post.
        mx_fila_post= x_{n1,0} x_{n1,1} .... x_{n1,nx}           my_fila_post= y_{n0,1}  y_{n1,1} ... y_{ny,1} 
                      x_{n2,0} x_{n2,1} .... x_{n2,nx}                         y_{n0,2}  y_{n1,2} ... y_{ny,2}
                         .                                                                   .        .
                         .                                                                   .        .
                      x_{nx,0} x_{nx,1} .... x_{nx,nx}                          y_{n0,ny} y_{n1,ny} ... y_{ny,ny}                             .                                                                   .        .
                      x_{nx,0} x_{nx,1} .... x_{nx,nx}                          y_{n0,ny} y_{n1,ny} ... y_{ny,ny} 
         
        d2=np.sqrt(np.power((mx_fila_post - malla_x), 2)+np.power((my_fila_post - malla_y), 2) )[0:n,0:n]
        
        
      Ara tenim 2 matrius, d1 i d2, amb les distàncies entre punts consecutius de la malla.
      
      Ara el que m'interesa d'aquestes distàncies es que no hi hagi molta desviació (és a dir que no hi hagi distàncies 
      de 35 i altres de 1)
    '''
    mx_col_post = np.copy(malla_x)
    my_col_post = np.copy(malla_y)
    mx_col_post[:,0:nx] = malla_x[:,1:(nx+1)]
    my_col_post[:, 0:ny] = malla_y[:, 1:(ny + 1)]

    mx_fila_post = np.copy(malla_x)
    my_fila_post = np.copy(malla_y)
    mx_fila_post[0:nx] = malla_x[1:(nx+1)]
    my_fila_post[0:ny] = malla_y[1:(ny + 1)]

    n = min(nx, ny)

    d1 = np.sqrt(np.power((mx_col_post - malla_x), 2) + np.power((my_col_post - malla_y), 2))[0:n, 0:n]
    d2 = np.sqrt(np.power((mx_fila_post - malla_x), 2) + np.power((my_fila_post - malla_y), 2))[0:n, 0:n]


    sd1 = np.std(d1)
    sd2 = np.std(d2)
    d1=np.mean(d1)-delta[1]
    d2=np.mean(d2)-delta[0]


    '''
    #per a regularitzar:
    mpost_x = np.copy(malla_x)
    mpost_x[0:nx] = malla_x[1:(nx+1)]
    rx = np.abs(np.mean(np.abs(mpost_x[0:nx]-malla_x[0:nx])) - delta[0])
    sdx = np.std(np.mean(np.abs(mpost_x[0:nx]-malla_x[0:nx])))

    mpost_y = np.copy(malla_y)
    mpost_y[:,0:ny] = malla_y[:,1:(ny+1)]
    ry = np.abs(np.mean(np.abs(mpost_y[:,0:ny]-malla_y[:,0:ny])) - delta[1])
    sdy = np.std(np.mean(np.abs(mpost_y[:,0:ny]-malla_y[:,0:ny])))

    #alternativa regularitzar
    n = min(nx,ny)
    R = (np.power(mpost_x[0:n, 0:n] - malla_x[0:n, 0:n], 2) + np.power(mpost_y[0:n, 0:n] - malla_y[0:n, 0:n],2))
    R = np.sqrt(R)
    Res = np.mean(R)
    s=np.std(R)

    r = np.sum(np.abs(parametres-mallav_original))
    r= np.power(parametres-mallav_original, 2)
    r= np.sum(r)
    r = np.sqrt (r)
    N = parametres.size
    r = np.sum(r)/N
    '''
    print(dif2, (sd1+sd2))

    #return dif2 + 10 *(rx + ry + sdx + sdy)
    return dif2 +(sd1+sd2)#+(d1+d2)
    #return dif.flatten()

#2a opcio
def min_colors_transform_nearest_neighbours(parametres):

    malla_x = parametres[0:(nx+1)*(ny +1)].reshape((nx+1),(ny +1))
    malla_y = parametres[(nx+1)*(ny+1): 2*(nx+1)*(ny+1)].reshape((nx+1),(ny +1))

    x = np.arange(imatge_input.shape[0])
    y = np.arange(imatge_input.shape[1])

    Coord_originals_x, Coord_originals_y = np.meshgrid(x, y)
    Coord_originals_x = Coord_originals_x.ravel()
    Coord_originals_y = Coord_originals_y.ravel()

    Coordenades_desti = posicio(Coord_originals_x, Coord_originals_y, malla_x, malla_y)
    imatge_registrada = colors_transform_nearest_neighbours(imatge_reference, Coordenades_desti)


    dif = np.power(imatge_input - imatge_registrada, 2)
    dif = np.sum(dif,2)
    dif = np.sqrt (dif)
    N = (imatge_input.shape[0])*(imatge_input.shape[1])
    dif = np.sum(dif)/N





        #cas filtratge gaussià:

    dif2 = np.power(imatge_input_gaussian-imatge_registrada, 2)
    dif2 = np.sum(dif2,2)
    dif2 = np.sqrt (dif2)
    N = (imatge_input.shape[0])*(imatge_input.shape[1])
    dif2 = np.sum(dif2)/N

    mx_col_post = np.copy(malla_x)
    my_col_post = np.copy(malla_y)
    mx_col_post[:,0:nx] = malla_x[:,1:(nx+1)]
    my_col_post[:, 0:ny] = malla_y[:, 1:(ny + 1)]

    mx_fila_post = np.copy(malla_x)
    my_fila_post = np.copy(malla_y)
    mx_fila_post[0:nx] = malla_x[1:(nx+1)]
    my_fila_post[0:ny] = malla_y[1:(ny + 1)]

    n = min(nx, ny)

    d1 = np.sqrt(np.power((mx_col_post - malla_x), 2) + np.power((my_col_post - malla_y), 2))[0:n, 0:n]
    d2 = np.sqrt(np.power((mx_fila_post - malla_x), 2) + np.power((my_fila_post - malla_y), 2))[0:n, 0:n]


    sd1 = np.std(d1)
    sd2 = np.std(d2)
    d1=np.mean(d1)-delta[1]
    d2=np.mean(d2)-delta[0]

    '''
    #per a regularitzar:
    mpost_x = np.copy(malla_x)
    mpost_x[0:nx] = malla_x[1:(nx+1)]
    rx = np.abs(np.mean(np.abs(mpost_x[0:nx]-malla_x[0:nx])) - delta[0])
    sdx = np.std(np.mean(np.abs(mpost_x[0:nx]-malla_x[0:nx])))

    mpost_y = np.copy(malla_y)
    mpost_y[:,0:ny] = malla_y[:,1:(ny+1)]
    ry = np.abs(np.mean(np.abs(mpost_y[:,0:ny]-malla_y[:,0:ny])) - delta[1])
    sdy = np.std(np.mean(np.abs(mpost_y[:,0:ny]-malla_y[:,0:ny])))
    #cas filtratge gausià:

    dif2 = np.power(imatge_registrada-imatge_reference_gaussian, 2)
    dif2 = np.sum(dif2,2)
    dif2 = np.sqrt (dif2)
    N = (imatge_input.shape[0])*(imatge_input.shape[1])
    dif2 = np.sum(dif2)/N

    #alternativa regularitzar
    n = min(nx,ny)
    R = (np.power(mpost_x[0:n, 0:n] - malla_x[0:n, 0:n], 2) + np.power(mpost_y[0:n, 0:n] - malla_y[0:n, 0:n],2))
    R = np.sqrt(R)
    Res = np.mean(R)
    s=np.std(R)

    r = np.sum(np.abs(parametres-mallav_original))
    r= np.power(parametres-mallav_original, 2)
    r= np.sum(r)
    r = np.sqrt (r)
    N = parametres.size
    r = np.sum(r)/N

    #terme regularitzador antic
    
    r = np.sum(np.abs(parametres-mallav_original))
    r= np.power(parametres-mallav_original, 2)
    r= np.sum(r)
    r = np.sqrt (r)
    N = parametres.size
    r = np.sum(r)/N

    '''
    print(dif2, 0.2*(sd1+sd2))
    return  dif2 + 0.2*(sd1+sd2)
    #tfi = time()
    # print('temps',tfi-titer )

#info mutua no funciona
def min_info_mutua(parametres):
    malla_x = parametres[0:(nx+1)*(ny +1)].reshape((nx+1),(ny +1))
    malla_y = parametres[(nx+1)*(ny+1): 2*(nx+1)*(ny+1)].reshape((nx+1),(ny +1))

    x = np.arange(imatge_input.shape[0])
    y = np.arange(imatge_input.shape[1])

    Coord_originals_x, Coord_originals_y = np.meshgrid(x, y)
    Coord_originals_x = Coord_originals_x.ravel()
    Coord_originals_y = Coord_originals_y.ravel()

    Coordenades_desti = posicio(Coord_originals_x, Coord_originals_y, malla_x, malla_y)
    #imatge_registrada = imatge_transformada(imatge_input, Coordenades_desti)
    R = colors_transform_nearest_neighbours(imatge_reference, Coordenades_desti)

    from spline_registration.losses import info_mutua
    #info = info_mutua(imatge_registrada,imatge_reference,7)
    info = info_mutua(R, imatge_input, 7)
    return -info

from scipy.optimize import least_squares
#topti=time()
#resultatlm = least_squares(min_imatge_transformada, malla_vector, method='lm')
#tfi=time()
#print(resultatlm, tfi-topti)


topti=time()
resultat_opcio1= least_squares(min_imatge_transformada, malla_vector, diff_step=0.1, method='trf',verbose=2)
tfi=time()
print(resultat_opcio1, tfi-topti)




'''
topti=time()
resultat_opcio2= least_squares(min_colors_transform_nearest_neighbours, malla_vector, diff_step=0.1, method='trf',verbose=2)
tfi=time()
print(resultat_opcio2, tfi-topti)
'''




'''
topti=time()
resultat_opcio31= least_squares(min_info_mutua, malla_vector, diff_step=0.1, method='trf',verbose=2)
tfi=time()
print(resultat_opcio31, tfi-topti)
'''



'''
topti=time()
resultattrf = least_squares(min_imatge_transformada, malla_vector, diff_step=0.18, method='trf',verbose=2)
tfi=time()
print(resultattrf, tfi-topti)
'''


'''
topti=time()
resultat = least_squares(min_colors_transform_nearest_neighbours, malla_vector, diff_step=0.1, method='trf', verbose=2, max_nfev=1)
tfi=time()
print(resultat, tfi-topti)

'''


'''
LA INFORMACIO MUTUA NO FUNCIONA!!!
topti=time()
resultatinfo = least_squares(min_info_mutua, malla_vector,diff_step=0.18, method='trf', verbose=2, max_nfev=10)
tfi=time()
print(resultatinfo, tfi-topti)
'''








parametres_optims = resultat_opcio1.x


malla_x = parametres_optims[0:(nx+1)*(ny+1)].reshape((nx+1),(ny+1))
malla_y = parametres_optims[(nx+1)*(ny+1):2*(nx+1)*(ny+1)].reshape((nx+1),(ny+1))


x = np.arange(imatge_input.shape[0])
y = np.arange(imatge_input.shape[1])

Coord_originals_x, Coord_originals_y  = np.meshgrid(x, y)
Coord_originals_x = Coord_originals_x.ravel()
Coord_originals_y  = Coord_originals_y .ravel()
Coordenades_desti = posicio(Coord_originals_x, Coord_originals_y, malla_x, malla_y)


imatge_registrada = imatge_transformada(imatge_input, Coordenades_desti)#opcio1
#imatge_registrada = colors_transform_nearest_neighbours(imatge_reference, Coordenades_desti) #opcio2

#guardar els resultats a una carpeta per cada experiment


from spline_registration.utils import create_results_dir
path_carpeta_experiment = create_results_dir('corregistre_ca')

fitxer_sortida = open(f'{path_carpeta_experiment}/descripció prova feta.txt', "w+")
fitxer_sortida.write(f'malla de dimensions: nx={nx}, ny ={ny}\n la imatge corregistrada lhem obtinguda utilitzant '
                     f'la funció que minimitz: min_imatge_transformada'
                     f'amb error quatràtic'
                     f'\n a naquest cas hem aplicat un filtratge gaussià a la imatge_input abans de res.'
                     f'\n el filtratge gaussià té sigma={sigma}.'
                     f'\n el temps tardat per fer loptimització dels parametres és de: {tfi-topti} segons.'
                     f'\n diff_step=0.1'
                     f'\n el resultat de la optimització: {resultat_opcio1}'
                     f'\n terme regularitzador, a partir de les distàncies entre valors consecutius a la malla, el que faig és afegir la desviació entre les distàncies aixó tendeix a ser 0 quam hi ha menys desviació ')
fitxer_sortida.close()


path_imatge_input = f'{path_carpeta_experiment}/imatge_input.png'
path_imatge_reference = f'{path_carpeta_experiment}/imatge_reference.png'
path_imatge_registrada = f'{path_carpeta_experiment}/imatge_registrada.png'

imsave(path_imatge_input,imatge_input)
imsave(path_imatge_reference,imatge_reference)
imsave(path_imatge_registrada,imatge_registrada)



#gif amb les 2 imatges la registrada i la de referència
path_gif = f'{path_carpeta_experiment}/movie.gif'
import imageio
images=[imatge_registrada, imatge_reference]
imageio.mimwrite(path_gif, images, fps=1)



#imatges superposades
    #a color
#imatge_superposada_color = imatge_registrada/256 * 0.5 + imatge_reference/256 * 0.5
#path_imatge_superposada= f'{path_carpeta_experiment}/imatge_superposada_color.png'
#imsave(path_imatge_superposada,imatge_superposada_color)

    #amb escala de grisos
from PIL import Image
im1 = Image.open(path_imatge_reference).convert('L')
im2 = Image.open(path_imatge_registrada).convert('L')
im = Image.blend(im1, im2, 0.5)
path_imatge_blend = f'{path_carpeta_experiment}/imatge_blend.png'
im = im.save(path_imatge_blend)

    #mapa de color

# Colorejar imatges en blanc i negre:

#cmap = plt.cm.get_cmap('winter')  # 'bone', 'summer', 'wistia', 'blues', 'reds', ...
#cmap2 = plt.cm.get_cmap('gray')

#imatge_reference_cmap = cmap(np.asarray(im1))
#imatge_registrada_cmap = cmap2(np.asarray(im2))

#path_reference_cmap = f'{path_carpeta_experiment}/reference_cmap.png'
#imsave(path_reference_cmap,imatge_reference_cmap)

#path_registrada_cmap = f'{path_carpeta_experiment}/registrada_cmap.png'
#imsave(path_registrada_cmap,imatge_registrada_cmap)

#gris = imread(path_registrada_cmap )
#color= imread(path_reference_cmap )
#imatge_superposada_cmap = gris/256 *0.5 + color/256 *0.5
#plt.imshow(imatge_superposada_cmap)
#plt.show()






import pylab as pl
#VEURE MALLA SOBRE LA IMATGE IMPUT

path_malla_inicial = f'{path_carpeta_experiment}/malla_inicial.png'
plt.imshow(imatge_input)
pl.plot(malla[0],malla[1],color='blue')
pl.plot(malla[1],malla[0],color='blue')
pl.title('malla inicial sobre la imatge input')
pl.savefig(path_malla_inicial)



#VEURE A ON VA A PARAR LA MALLA I AMB QUINS VALORS HO COMPARA

path_malla_transformada = f'{path_carpeta_experiment}/malla_transformada.png'
plt.imshow(imatge_registrada)
pl.plot(malla_x,malla_y,color='green')
pl.plot(malla_y,malla_x,color = 'green')
pl.title('malla òptima sobre la imatge registrada')
pl.savefig(path_malla_transformada)




'''
from skimage.filters import gaussian
im_gaussian = gaussian(imatge_reference, sigma=5, multichannel=True)

def min_gaussian_SSD(parametres):
    malla_x = parametres[0:nx * ny].reshape(nx, ny)
    malla_y = parametres[nx * ny:2 * nx * ny].reshape(nx, ny)

    x = np.arange(coordenadesx[nx - 1] - 1)
    y = np.arange(coordenadesy[ny - 1] - 1)
    Coord_originals_x, Coord_originals_y = np.meshgrid(x, y)
    Coord_originals_x = Coord_originals_x.ravel()
    Coord_originals_y = Coord_originals_y.ravel()
    Coordenades_desti = posicio(Coord_originals_x, Coord_originals_y, malla_x, malla_y)

    imatge_registrada = imatge_transformada(imatge_input, Coordenades_desti)

    from spline_registration.losses import SSD
    info = SSD(imatge_registrada,im_gaussian)
    return info

res = least_squares(min_gaussian_SSD,malla_vector, diff_step=0.18,method='trf',verbose=2)
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
visualize_side_by_side(imatge_registrada,imatge_reference,'registrada i reference')
'''
