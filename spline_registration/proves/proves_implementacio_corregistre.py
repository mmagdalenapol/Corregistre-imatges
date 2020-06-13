'''
Farem les proves amb la imatge del moix així ja que la dimenció es menor i no feim a l'ordenador fer més feina de la necessària.
'''
import numpy as np
import matplotlib.pyplot as plt
from skimage import data
imatge = data.chelsea()
'''
el primer que ens plantegem és com fer una malla. He fet diverses proves fins arribar a decidir quina era més fàcil per jo fer feina. 
1r definim la resolució de la malla nx,ny.
'''
#volem fer una malla de 20x20
nx, ny = (20,20)

'''
apartir d'aquí coses que no

    malla1 = np.mgrid[0:imatge1.shape[0]:int(imatge1.shape[0]/nx), 0:imatge1.shape[1]:int(imatge1.shape[1]/ny)]
    malla2=np.ogrid[0:imatge1.shape[0]:int(imatge1.shape[0]/nx), 0:imatge1.shape[1]:int(imatge1.shape[1]/ny)]
    x = np.arange(0, imatge1.shape[0], imatge1.shape[0]/nx)
    y = np.arange(0, imatge1.shape[1], imatge1.shape[1]/ny)
    malla = np.meshgrid(x, y, sparse=True)

    #perque siguin enters
    malla[0]=np.floor(malla[0]).astype(int)
    malla[1]=np.floor(malla[1]).astype(int)
    #malla és un np array de manera que malla[0] té les coordenades x de la malla i malla[1] les coodenades y.
    #per a veure la malla visualment

    size=(imatge1.shape[0],imatge1.shape[1])
    base = np.copy(imatge1)
    base[0:imatge1.shape[0],0:imatge1.shape[1]]=[255,255,255]
    base[malla[0],malla[1]]=[255,0,0]
    #per fer la quadricula
    base[malla[0],:]=[0,255,0]
    base[:,malla[1]]=[0,255,0]
    plt.imshow(base)
    plt.show()

    #dibuixa els punts de la malla ho fa bé però com n'hi ha tants no se veu bé


    rotate = ndimage.rotate(imatge1, 30)
    base_rotate = ndimage.rotate(base, 30)
    rotate_noreshape = ndimage.rotate(imatge1, 30, reshape=False)

    
    
    #vull una matriu que em digui cada punt de la malla a on me les envia M
    
    
    M = np.empty([imatge.shape[0],imatge.shape[1],2])

    from spline_registration.transform_models import trans_prova

    transformada_rotacio = trans_prova()
    imatge_rotada = transformada_rotacio.apply_transform(imatge,imatge)

    x = np.arange(0, imatge1.shape[0], imatge1.shape[0]/nx)
    y = np.arange(0, imatge1.shape[1], imatge1.shape[1]/ny)
    malla = np.meshgrid(x, y, sparse=True)
    #perque siguin enters
    malla[0]=np.floor(malla[0]).astype(int)
    malla[1]=np.floor(malla[1]).astype(int)



fins aquí les coses que no 
'''

'''
Anem a començar amb un cas molt bàsic una rotació per a la Transformada
'''
def rotacio(x0,y0,xc,yc, alpha):
    x = xc+(x0-xc )* np.cos(alpha) - (y0-yc) * np.sin(alpha)
    y = yc+ (x0-xc) * np.sin(alpha) + (y0-yc) * np.cos(alpha)
    return(x,y)
nx, ny = (20,20)
malla = np.mgrid[0:imatge.shape[0]:round(imatge.shape[0]/nx), 0:imatge.shape[1]:round(imatge.shape[1]/ny)]

'''
malla ens torna un narray 
malla[0] té 20 files i 20 columnes però totes ses columnes són iguals i ens diu les files que tenim en compte de cada columna
malla[1] té 20 files i 20 columnes però totes les files són iguals i ens diu les columnes que tenim en compte de cada fila.
'''
coordenadesx = malla[0][:,0]
coordenadesy = malla[1][0,:]
delta = [malla[0][1,0],malla[1][0,1]]
#si no començassim amb el (0,0) delta = [malla[0][1,0]-malla[0][0,0],malla[1][0,1]-malla[1][0,0] ]
alpha = np.pi/4
asignacio = {(x,y):rotacio(x,y,0,0, alpha) for x in coordenadesx for y in coordenadesy}


def Bspline(t):
    B = np.zeros(4)
    B[0] = ((1-t)**3)/6
    B[1] = (3*(t**3) - 6*(t**2) + 4)/6
    B[2] = (-3*(t**3) + 3*t**2 + 3*t + 1 )/6
    B[3] = (t**3)/6
    return B

'''
s=[0.0,0.06666666666666667,0.13333333333333333,0.2,0.26666666666666666,0.3333333333333333,0.4,0.4666666666666667,0.5333333333333333,0.6,0.6666666666666666,0.7333333333333333,0.8,0.8666666666666667,0.9333333333333333,0.0]

t=[0.0, 0.043478260869565216, 0.08695652173913043, 0.13043478260869565, 0.17391304347826086, 0.21739130434782608, 0.2608695652173913, 0.30434782608695654, 0.34782608695652173, 0.391304347826087, 0.43478260869565216, 0.4782608695652174, 0.5217391304347826, 0.5652173913043478, 0.6086956521739131, 0.6521739130434783, 0.6956521739130435, 0.7391304347826086, 0.782608695652174, 0.8260869565217391, 0.8695652173913043, 0.9130434782608695, 0.9565217391304348]

'''
def posicio(x,y):
    f=[0,0]
    s = x / delta[0] - np.floor(x / delta[0])
    t = y / delta[1] - np.floor(y / delta[1])
    i = np.floor(x / delta[0]).astype(int)
    j = np.floor(y / delta[1]).astype(int)

    for k in range (0,4):
        for l in range (0,4):
            f = f + Bspline(s)[k] * Bspline(t)[l] * np.array(
                [asignacio[(i+(k-1))*delta[0], (j+(l-1))*delta[1]][0],
                 asignacio[(i+(k-1)) * delta[0], (j+(l-1))* delta[1]][1]])
    return f




T = np.empty((imatge.shape[0],imatge.shape[1],2))
for x in range (delta[0],coordenadesx[(nx-2)]):
    for y in range (delta[1],coordenadesy[(ny-2)]):
        T[x,y]=posicio(x,y)


'''
no te sentit mirar posicio(0,0) ni (0,x) o (x,0) ja que utilitza el punt d'una posició abans.
  posicio(15,23)=asignacio[15,23],
  posicio(15,46)=asignacio[15,46].
  .
  .
  .
  posicio(255,391)=asignacio[255,391] es la darrera posicio que podem mirar ja que utilitzam la asignació dels 2 posteriors.
  posicio(269,413) (segueix tenint dos punts de control abans)
  
'''

X=T[:,:,0]
Y=T[:,:,1]
from spline_registration.utils import imatge_vec
Xvec= imatge_vec(X,1)
Yvec= imatge_vec(Y,1)

min=abs(np.min(T)) 
V=np.ones_like(T)
V= T+min*V
VX=V[:,:,0]
VY=V[:,:,1]
VXvec= imatge_vec(VX,1)
VYvec= imatge_vec(VY,1)
#dim=np.ceil(abs(np.max(T))+abs(np.min(T))).astype(int)
dim=np.ceil(abs(np.max(V))+abs(np.min(V))).astype(int)
quadricula=np.zeros((dim,dim,3))
quadricula[VXvec.astype(int),VYvec.astype(int)]=[0,255,0]
plt.imshow(quadricula)
plt.show()


from spline_registration.transform_models import trans_prova
transformada_rotacio = trans_prova()
imatge_rotada = transformada_rotacio.apply_transform(imatge, imatge)
imatge_rotada[VXvec.astype(int)[0:100],VYvec.astype(int)[0:100]]=[255,0,0]
plt.imshow(quadricula)
plt.show()


#nova versio 15 de maig

import numpy as np
from skimage.io import imread
from spline_registration.transform_models import Rescala
from skimage import data
imatge = data.chelsea()


imatge_input=imread('/Users/mariamagdalenapolpujadas/Desktop/universitat/tfg/GITHUB/spline_registration/proves/dog_input.jpg')
imatge_reference= imread('/Users/mariamagdalenapolpujadas/Desktop/universitat/tfg/GITHUB/spline_registration/proves/dog_reference.jpg')
nx, ny = (20,20)
malla = np.mgrid[0:imatge_reference.shape[0]:round(imatge_reference.shape[0]/nx), 0:imatge_reference.shape[1]:round(imatge_reference.shape[1]/ny)]
coordenadesx = np.arange(0,imatge_reference.shape[0],round(imatge_reference.shape[0]/nx))
coordenadesy = np.arange(0,imatge_reference.shape[1],round(imatge_reference.shape[1]/ny))
malla_x = malla[0]#inicialitzam a on van les coordenades x
malla_y = malla[1]#inicialitzam a on van les coordenades y
 #ara utilitz interpolacio lineal.
def posicio(x,y):

    s = x / delta[0] - np.floor(x / delta[0]) #val 0 quan la x està a coordenadesx
    t = y / delta[1] - np.floor(y / delta[1]) #val 0 quan la y està a coordenadesy
    i = np.floor(x / delta[0]).astype(int) #index de la posició més pròxima per davall de la coordenada x a la malla
    j = np.floor(y / delta[1]).astype(int) #index de la posició més pròxima per davall de la coordenada y a la malla

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

#transformada=Rescala()
#dim=transformada.find_best_transform(imatge_reference,imatge_input)#dimensio de la imatge que volem reescalar
#imatge_rotada = transformada.apply_transform(imatge,imatge_rotada)#rescalam la imatge 2 per tal que sigui de la mateixa dimensio que la 1

A = malla # coordenades d'on se suposa que ve cada pixel de la imatge corregistrada.
Ax = A[0].ravel()
Ay = A[1].ravel()
#A = np.array([Ax,Ay])
A=np.concatenate((Ax,Ay ), axis=0)


A2 =np.array([malla_x.ravel(),malla_y.ravel()])#coordenades a la imatge 2
A2x=malla_x.ravel()
A2y=malla_y.ravel()
p=np.concatenate((A2x,A2y ), axis=0) #array_like with shape (n,) on n= nx*ny*2
#a p tenim a les primeres nx*ny columnes les coordenades x de la malla i a les segones les coordenades y.


def colors_transform(reference_image, A):
    '''
    A
    la transformació m'hauria de donar les coordenades d'on se suposa que ve cada pixel de la imatge corregistrada.
    Aquestes coordenades poden ser decimals i per això el valor del color depèn del color dels 4 pixels més propers.
    A encara és una matriu

    Abans de res feim uns quantes adaptacions inicials a A.
    -No hi ha coordenades negatives per tant qualsevol element negatiu el passam a nes 0.
    -La coordenada x no pot ser major que les files de la imatge de referència.
    -La coordenada y no pot ser major que les columnnes de la imatge de referència.
    '''
    A= np.array([A[0:(nx*ny)],A[nx*ny:(2*nx*ny)]])
    #si li introduim A com un array_like with shape (n,) on n= nx*ny*2 feim aquest canvi per emprar el mateix codi.

    A = np.maximum(A, 0)
    A[0] = np.minimum(A[0], reference_image.shape[0] - 1)
    A[1] = np.minimum(A[1], reference_image.shape[1] - 1)

    columna1 = A[0]
    columna2 = A[1]

    '''
    A11 A12
    A21 A22

    per tant:
     A11: és la x floor i la y ceil      A12: és la x i la y ceil 
     A21: és la x i la y floor           A22: és la x ceil i la y floor
    '''

    A11 = np.ones(A.shape)
    A11[0] = np.floor(columna1)
    A11[1] = np.ceil(columna2)

    A12 = np.ceil(A)

    A21 = np.floor(A)

    A22 = np.ones(A.shape)
    A22[0] = np.ceil(columna1)
    A22[1] = np.floor(columna2)


    '''
    f11,f12,f21 i f22 són els coeficients per la interpolació bilineal. 
    és a dir color(x,y) = color(A21(x,y))*f21(x,y) + color(A22(x,y))*f22(x,y) + color(A11(x,y))*f11(x,y) + color(A12(x,y))*f12(x,y)

    així inicialment les possam tots 0 i les anam canviant segons si com sigui (x,y). 
    '''
    # A es (2,400,1)
    f11 = np.zeros(A.shape[1])  # només necessitam un nombre per posició
    f12 = np.zeros(A.shape[1])
    f21 = np.zeros(A.shape[1])
    f22 = np.zeros(A.shape[1])

    '''
    denx,deny,num21x,num21y,num11y,num22x ens serviràn per escriuré més fàcilment qui són f11,f12,f21 i f22 en cada cas
    '''
    denx = (A12[0] - A21[0])
    deny = (A12[1] - A21[1])

    num21x = A12[0] - A[ 0]
    num21y = A12[1] - A[1]
    num11y = A21[1] - A[1]
    num22x = A21[0] - A[0]

    '''
    inicialment totes són 0 i segons quina situació estam ho anirem canviant.

    Si la coordenada (x,y) ja eren els dos nombres enters llavors ens trobam al cas (x,y) = (np.ceil(x), np.ceil(y)) 
    i per tant si volem trobar els pixels que satisfan aquesta condició el que volem és trobar aquelles posicions tals que 
    A==A21. Això es equivalent als llocs on denx + deny == 0). 

    El cas en què amdbues coordenades són decimals és equivalent a dir denx+deny==2. 
    per tant en aquesta situació hem de canviar tots els valors.

    Cas x decimal i y entera equival a deny=0 i denx=1. Per tant en aquest cas els afectats són:  deny - denx = -1
    Cas x enter i y decimal equival a deny=1 i denx=0. Per tant equival a: deny - denx = 1
    '''
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

imatge_malla=colors_transform(imatge_reference, A)
imatge_malla=np.hsplit(imatge_malla, nx*ny)
imatge_malla=np.array(imatge_malla)
imatge_malla=(imatge_malla.ravel()).reshape(nx,ny,3)


imatge_malla2= colors_transform(imatge_input,A2)
imatge_malla2=np.hsplit(imatge_malla2, nx*ny)
imatge_malla2=np.array(imatge_malla2)
imatge_malla2=(imatge_malla2.ravel()).reshape(nx,ny,3)






def min(p):

    imatge_p = colors_transform(imatge_input,p)
    imatge_p=np.hsplit(imatge_p, nx*ny)
    imatge_p=np.array(imatge_p)
    imatge_p=(imatge_p.ravel()).reshape(nx,ny,3)

    from spline_registration.losses import SSD
    return SSD(imatge_malla,imatge_p)

from scipy.optimize import least_squares
res = least_squares(min, p,method='lm' )
parametres = res.x

parametres = np.array([parametres[0:(nx * ny)], parametres[nx * ny:(2 * nx * ny)]])
