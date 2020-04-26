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
malla = np.mgrid[0:imatge.shape[0]:int(imatge.shape[0]/nx), 0:imatge.shape[1]:int(imatge.shape[1]/ny)]

'''
malla ens torna un narray 
malla[0] té 20 files i 20 columnes però totes ses columnes són iguals i ens diu les files que tenim en compte de cada columna
malla[1] té 20 files i 20 columnes però totes les files són iguals i ens diu les columnes que tenim en compte de cada fila.
'''
coordenadesx = malla[0][:,0]
coordenadesy = malla[1][0,:]
alpha = np.pi/4
asignacio = {(x,y):rotacio(x,y,150,226, alpha) for x in coordenadesx for y in coordenadesy}

