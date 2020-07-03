import numpy as np
from skimage.io import imread
from time import time


imatge_input = imread('/Users/mariamagdalenapolpujadas/Desktop/universitat/tfg/GITHUB/spline_registration/proves/dog_input.jpg')
imatge_reference = imread('/Users/mariamagdalenapolpujadas/Desktop/universitat/tfg/GITHUB/spline_registration/proves/dog_reference.jpg')

nx, ny = (20,20)
malla = np.mgrid[0:imatge_input.shape[0]:round(imatge_input.shape[0]/nx), 0:imatge_input.shape[1]:round(imatge_input.shape[1]/ny)]

coordenadesx = np.arange(0,imatge_input.shape[0],round(imatge_input.shape[0]/nx))
coordenadesy = np.arange(0,imatge_input.shape[1],round(imatge_input.shape[1]/ny))
delta = [coordenadesx[1],coordenadesy[1]]
malla_x = malla[0]#inicialitzam a on van les coordenades x a la imatge_reference
malla_y = malla[1]#inicialitzam a on van les coordenades y a la imatge_reference

'''
Alternativa per trobar A2:
A2x=malla_x.ravel()
A2y=malla_y.ravel()
A2=np.concatenate((A2x,A2y ), axis=0) #array_like with shape (n,) on n= nx*ny*2
print('a A2 tenim a les primeres nx*ny columnes les coordenades x de la malla i a les segones les coordenades y. Fent referencia a coordenades de la imatge reference \n')
'''
A2 = malla.flatten()
#a A2 tenim a les primeres nx*ny columnes les coordenades x de la malla i a les segones les coordenades y.

#com inicialitzam A2 per a que sigui la identitat tenim: A=A2 són les posicións de la malla a la imatge input també.
A = np.copy(A2)
print('com inicialitzam A2 per a que sigui la identitat tenim: A=A2 són les posicións de la malla a la imatge input també.')

def colors_transform(reference_image,A):

    '''
    #si li introduim A com un array_like with shape (n,) on n= nx*ny*2.
    #ara A es és una matriu de dimensió (2, nx*ny)
    # A[0] conté les coordenades x, A[1] conté les coordenades y.
    '''
    if A.shape == (2*nx*ny,):#per quan ficam la malla A2
        A = np.array([A[0:(nx * ny)], A[nx * ny:(2 * nx * ny)]])
        '''
        #si li introduim A com un array_like with shape (n,) on n= nx*ny*2.
        #ara A es és una matriu de dimensió (2, nx*ny)
        # A[0] conté les coordenades x, A[1] conté les coordenades y.
        '''
    A = np.maximum(A, 0)

    A[0] = np.minimum(A[0], reference_image.shape[0] - 1)
    A[1] = np.minimum(A[1], reference_image.shape[1] - 1)


    X = A[0]
    Y = A[1]
    Ux = np.floor(X).astype(int)
    Vy = np.floor(Y).astype(int)

    a = X-Ux
    b = Y-Vy

    #si a és 0 no hauriem de tenir en compte Ux+1 tan sols Ux i si b es 0 no he m de tenir en compte Vy+1 tan sols Vy
    M = np.array([reference_image[Ux, Vy][:, 0],reference_image[Ux, Vy][:, 1],reference_image[Ux, Vy][:, 2]])
    B = np.array([reference_image[Ux + 1, Vy][:,0],reference_image[Ux + 1, Vy][:,1],reference_image[Ux + 1, Vy][:,2]])
    C = np.array([reference_image[Ux, Vy + 1][:,0],reference_image[Ux, Vy + 1][:,1],reference_image[Ux, Vy + 1][:,2]])
    D = np.array([reference_image[Ux+1, Vy+1][:,0],reference_image[Ux+1, Vy+1][:,1],reference_image[Ux+1, Vy+1][:,2]])


    color = (a-1)*(b-1)*M + a*(1-b)*B + (1-a)*b*C + a*b*D

    return color


'''
TAMBÉ FUNCIONA PERÒ ÉS MÉS COMPLICAT PER TANT HEM QUED AMB LA VERSIO SIMPLE
def colors_transform(reference_image, A):
    
   # A
    #la transformació m'hauria de donar les coordenades d'on se suposa que ve cada pixel de la imatge corregistrada.
    #Aquestes coordenades poden ser decimals i per això el valor del color depèn del color dels 4 pixels més propers.
    #A encara és una matriu

    #Abans de res feim uns quantes adaptacions inicials a A.
    #-No hi ha coordenades negatives per tant qualsevol element negatiu el passam a nes 0.
    #-La coordenada x no pot ser major que les files de la imatge de referència.
    #-La coordenada y no pot ser major que les columnnes de la imatge de referència.
    
    A= np.array([A[0:(nx*ny)],A[nx*ny:(2*nx*ny)]])
    #si li introduim A com un array_like with shape (n,) on n= nx*ny*2 feim aquest canvi per emprar el mateix codi.
    #ara A es és una matriu de dimensió (2, nx*ny)
    # A[0] conté les coordenades x, A[1] conté les coordenades y.

    A = np.maximum(A, 0)
    A[0] = np.minimum(A[0], reference_image.shape[0] - 1)
    A[1] = np.minimum(A[1], reference_image.shape[1] - 1)

    columna1 = A[0]
    columna2 = A[1]

    #A11 A12
    #A21 A22

    #per tant:
     #A11: és la x floor i la y ceil      A12: és la x i la y ceil 
     #A21: és la x i la y floor           A22: és la x ceil i la y floor


    A11 = np.ones(A.shape)
    A11[0] = np.floor(columna1)
    A11[1] = np.ceil(columna2)

    A12 = np.ceil(A)

    A21 = np.floor(A)

    A22 = np.ones(A.shape)
    A22[0] = np.ceil(columna1)
    A22[1] = np.floor(columna2)

    #f11,f12,f21 i f22 són els coeficients per la interpolació bilineal. 
    #és a dir color(x,y) = color(A21(x,y))*f21(x,y) + color(A22(x,y))*f22(x,y) + color(A11(x,y))*f11(x,y) + color(A12(x,y))*f12(x,y)

    #així inicialment les possam tots 0 i les anam canviant segons si com sigui (x,y). 

    # A es (2,400,1)
    f11 = np.zeros(A.shape[1])  # només necessitam un nombre per posició
    f12 = np.zeros(A.shape[1])
    f21 = np.zeros(A.shape[1])
    f22 = np.zeros(A.shape[1])

    #denx,deny,num21x,num21y,num11y,num22x ens serviràn per escriuré més fàcilment qui són f11,f12,f21 i f22 en cada cas

    denx = (A12[0] - A21[0])
    deny = (A12[1] - A21[1])

    num21x = A12[0] - A[0]
    num21y = A12[1] - A[1]
    num11y = A21[1] - A[1]
    num22x = A21[0] - A[0]


    #inicialment totes són 0 i segons quina situació estam ho anirem canviant.

    #Si la coordenada (x,y) ja eren els dos nombres enters llavors ens trobam al cas (x,y) = (np.ceil(x), np.ceil(y)) 
    #i per tant si volem trobar els pixels que satisfan aquesta condició el que volem és trobar aquelles posicions tals que 
    #A==A21. Això es equivalent als llocs on denx + deny == 0). 

    #El cas en què amdbues coordenades són decimals és equivalent a dir denx+deny==2. 
    #per tant en aquesta situació hem de canviar tots els valors.

    #Cas x decimal i y entera equival a deny=0 i denx=1. Per tant en aquest cas els afectats són:  deny - denx = -1
    #Cas x enter i y decimal equival a deny=1 i denx=0. Per tant equival a: deny - denx = 1

    f21[np.where(denx + deny == 0)] = 1

    den = np.where(denx + deny == 2, denx * deny, 1)  # per evitar problemes interns dividint entre 0

    f21[np.where(denx + deny == 2)] = ((num21x * num21y) / den)[np.where(denx + deny == 2)]
    f22[np.where(denx + deny == 2)] = ((num22x * num21y) / -den)[np.where(denx + deny == 2)]
    f11[np.where(denx + deny == 2)] = ((num21x * num11y) / -den)[np.where(denx + deny == 2)]
    f12[np.where(denx + deny == 2)] = ((num22x * num11y) / den)[np.where(denx + deny == 2)]


    denxmodificat = np.where(deny - denx == -1, denx, 1)  # per evitar problemes interns dividint entre 0
    f21[np.where(deny - denx == -1)] = (num21x / denxmodificat)[np.where(deny - denx == -1)]
    f22[np.where(deny - denx == -1)] = (num22x / -denxmodificat)[np.where(deny - denx == -1)]

    denymodificat = np.where(deny - denx == 1, deny, 1)  # per evitar problemes interns dividint entre 0
    f21[np.where(deny - denx == 1)] = (num21y / denymodificat)[np.where(deny - denx == 1)]
    f11[np.where(deny - denx == 1)] = (num11y / -denymodificat)[np.where(deny - denx == 1)]

    a11 = reference_image[A11[0].astype(int), A11[1].astype(int)]
    a12 = reference_image[A12[0].astype(int), A12[1].astype(int)]
    a21 = reference_image[A21[0].astype(int), A21[1].astype(int)]
    a22 = reference_image[A22[0].astype(int), A22[1].astype(int)]

    B = (f11*[a11[:,0],a11[:,1],a11[:,2]]+
         f12*[a12[:,0],a12[:,1],a12[:,2]]+
         f21 *[a21[:,0],a21[:,1],a21[:,2]]+
         f22 * [a22[:,0],a22[:,1],a22[:,2]])
    return B

'''

imatge_malla_input=colors_transform(imatge_input, A)
imatge_malla_input=np.hsplit(imatge_malla_input, nx*ny)
imatge_malla_input=np.array(imatge_malla_input)
imatge_malla_input=(imatge_malla_input.ravel()).reshape(nx,ny,3)

def funcio_minimitzar(p):

    imatge_p = colors_transform(imatge_reference,p)
    imatge_p=np.hsplit(imatge_p, nx*ny)
    imatge_p=np.array(imatge_p)
    imatge_p=(imatge_p.ravel()).reshape(nx,ny,3)



    from spline_registration.losses import SSD,info_mutua
    return SSD(imatge_malla_input,imatge_p)
    #return -info_mutua(imatge_malla_input,imatge_p,5)

def malla_optima(p):
    malla_x = p[0:nx*ny].reshape(nx,ny)
    malla_y = p[nx*ny:2*nx*ny].reshape(nx,ny)

    return (malla_x,malla_y)

from scipy.optimize import least_squares
#res = least_squares(funcio_minimitzar, A2,method='lm' )
#parametres = res.x


#malla_x = malla_optima(res.x)[0]
#malla_y = malla_optima(res.x)[1]
def posicio(x,y,malla_x,malla_y):


    s = x / delta[0] - np.floor(x / delta[0]) #val 0 quan la x està a coordenadesx
    t = y / delta[1] - np.floor(y / delta[1]) #val 0 quan la y està a coordenadesy
    i = np.floor(x / delta[0]).astype(int) #index de la posició més pròxima per davall de la coordenada x a la malla
    j = np.floor(y / delta[1]).astype(int) #index de la posició més pròxima per davall de la coordenada y a la malla

    '''
    AIXÒ ESTÀ MALAMENT
    xpost = coordenadesx[i + 1]
    xant = coordenadesx[i]
    ypost = coordenadesy[j + 1]
    yant = coordenadesy[j]
    if s!=0:
        posiciox= malla_x[i,j]*(xpost-x)/(xpost-xant) + malla_x[i+1,j+1]*(x-xant)/(xpost-xant)
    if t!=0:
        posicioy = malla_y[i,j] * (ypost - y) / (ypost - yant) + malla_y[i+1,j + 1] * (y - yant) / (ypost - yant)

    if s==0 :
        posiciox = malla_x[i,j]
    if t==0:
        posicioy = malla_y[i,j]

    return [posiciox,posicioy]
    '''
    A = np.array([malla_x[i, j], malla_y[i, j]])
    B = np.array([malla_x[i+1, j], malla_y[i+1, j]])
    C = np.array([malla_x[i, j+1], malla_y[i, j+1]])
    D = np.array([malla_x[i+1, j+1],malla_y[i+1,j+1]])
    interpolacio = (s-1)*(t-1)*A + s*(1-t)*B + (1-s)*t*C + s*t*D

    return interpolacio

'''
imatge_transformada = np.zeros((imatge_reference.shape[0],imatge_reference.shape[1],2))
for x in range(0, coordenadesx[(nx - 1)]):
    for y in range(0, coordenadesy[(ny - 1)]):
        imatge_transformada[x, y] = posicio(x, y)

'''


#1 de juliol

from spline_registration.utils import imatge_vec
I = imatge_vec(imatge_input[0:coordenadesx[(nx - 1)],0:coordenadesx[(ny - 1)]],3)
'''
Això és molt més lento millor emprar mgrid
T = np.zeros((I.shape[0],2))
i=0
tempsbucle=time()
for x in range(0,coordenadesx[(nx - 1)]):
    for y in range(0,coordenadesy[(ny - 1)]):
        T[i]=posicio(x,y)
        i=i+1
tempsfibucle=time()
'''


tempsgrid = time()
C = np.mgrid[0:coordenadesx[(nx - 1)], 0:coordenadesy[(ny - 1)]]
Cx = C[0].ravel()
Cy = C[1].ravel()
P=posicio(Cx,Cy,malla_x,malla_y)
#si vull la posició corresponent a I[j] hem de posar P[:,j]

tempsfigrid = time()

R=colors_transform(imatge_reference,P)
R=np.hsplit(R, coordenadesx[(nx - 1)]*coordenadesy[(ny - 1)])
R=np.array(R)
R=(R.ravel()).reshape(coordenadesx[(nx - 1)]*coordenadesy[(ny - 1)],3)
'''
R[:,1]== imatge_reference[P[:,1]] si tot son enters
R[:,1].astype(int)==imatge_reference[(P[:,1][0]).astype(int),(P[:,1][1]).astype(int)]
'''


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
    R = np.hsplit(R, coordenadesx[(nx - 1)] * coordenadesy[(ny - 1)])
    R = np.array(R)
    R = (R.ravel()).reshape(coordenadesx[(nx - 1)] * coordenadesy[(ny - 1)], 3)

    dif = np.sum(np.abs(I-R), 1)

    #dif = np.sum(I-R, 1)
    #dif_abs = np.abs(dif)
    return dif.flatten()

resultat = least_squares(funcio_min, A2, method='lm')
print(resultat)