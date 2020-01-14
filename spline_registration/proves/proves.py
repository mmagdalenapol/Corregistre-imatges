#from numpy import np
#def dim (imatge):
#   dimensions = reference_image.shape
#    dimensio_x = dimensions[0]
#    dimensio_y = dimensions[1]

#    pixel_vermell = reference_image[0][0][0]
#    reference_image - transformed_image




from skimage import io
from spline_registration.utils import get_databases_path

ic = io.ImageCollection(f'{get_databases_path()}/COAD01/*.jpg')#així carregam totes les imatges de la carpeta COAD01 que son jpg

#mostram per pantalla els noms d'on estan guardades les imatges
for i in range (0,len(ic)):
    print(ic.files[i])


#proves amb la funcio anhir i cerca_imatge_anhir
from spline_registration.databases import anhir
from spline_registration.databases import cerca_imatge_anhir
from spline_registration.utils import visualize_side_by_side
from skimage.io import imread
from spline_registration.losses import SSD
from spline_registration.losses import info_mutua
from spline_registration.transform_models import Rescala


imatge1 = imread(cerca_imatge_anhir (anhir(),'mammary-gland_2', 's2_64-PR_A4926-4L'))
imatge2 = imread(cerca_imatge_anhir (anhir(),'mammary-gland_2', 's2_62-ER_A4926-4L'))

transformada=Rescala()
dim=transformada.find_best_transform(imatge1,imatge2)#dimensio de la imatge que volem reescalar
imatge2 = transformada.apply_transform(imatge2,dim)#rescalam la imatge 2 per tal que sigui de la mateixa dimensio que la 1

visualize_side_by_side(imatge1,imatge2, 'Esquerra: s2_64-PR_A4926-4L.jpg ; Dreta: s2_62-ER_A4926-4L.jpg ')
print('L´error SSD entre les imatges de la mateixa mostra és:',SSD(imatge1, imatge2))
print('L´informació mútua entre les imatges de la mateixa mostra és:',info_mutua(imatge1, imatge2,5))
# com podem veure les 2 imatges són semblants però de colors diferents i per aquesta raó ja ens dóna un error molt gran.
# hem de trobar una mesura d'error millor.

imatge_mostra_diferent = imread(cerca_imatge_anhir (anhir(),'lung-lesion_1', '29-041-Izd2-w35-Cc10-5-les1'))
imatge_mostra_diferent = transformada.apply_transform(imatge_mostra_diferent,dim)


visualize_side_by_side(imatge1,imatge_mostra_diferent, 'Esquerra:mammary-gland_  ; Dreta: lung-lesion_1')
print('L´error SSD entre les imatges de mostres diferents és:',SSD(imatge1, imatge_mostra_diferent))
print('L´informació mútua entre les imatges de mostres diferents és:',info_mutua(imatge1, imatge_mostra_diferent))

#veim que ssd no ens aporta informacio util ja que ens diu que s'assemblen més imatges de mostres diferents que de la mateixa mostra.
#informació mutua en principi va millor ja que ens dóna menor un valor major en imatges de la mateixa mostra que en imatges de mostres diferents.


mammarygland_2_imatge1=imread(cerca_imatge_anhir (anhir(),'mammary-gland_2', 's2_61-HE_A4926-4L.jpg'))
mammarygland_2_imatge2=imread(cerca_imatge_anhir (anhir(),'mammary-gland_2', 's2_62-ER_A4926-4L.jpg'))
mammarygland_2_imatge3=imread(cerca_imatge_anhir (anhir(),'mammary-gland_2', 's2_63-HE_A4926-4L'))
mammarygland_2_imatge2= Rescala.find_best_transform(Rescala(),mammarygland_2_imatge1,mammarygland_2_imatge2)
mammarygland_2_imatge3= Rescala.find_best_transform(Rescala(),mammarygland_2_imatge1,mammarygland_2_imatge3)
print('informació mutua imatge 1 amb 2:',info_mutua(mammarygland_2_imatge1, mammarygland_2_imatge2),'informació mutua imatge 1 amb 3:',info_mutua(mammarygland_2_imatge1, mammarygland_2_imatge3),'informació mutua imatge 2 amb 3:',info_mutua(mammarygland_2_imatge2, mammarygland_2_imatge3))



#veim que la informació mutua es menor quan es tracta de imatges de mostres diferents.
imatge_mostra_diferent= imread(cerca_imatge_anhir (anhir(),'lung-lesion_1', '29-041-Izd2-w35-Cc10-5-les1'))
imatge_mostra_diferent= Rescala.find_best_transform(Rescala(),mammarygland_2_imatge1,imatge_mostra_diferent)
print('informació mutua imatge 1 amb diferent:', info_mutua(mammarygland_2_imatge1, imatge_mostra_diferent),
      'informació mutua imatge diferent amb 3:', info_mutua(imatge_mostra_diferent, mammarygland_2_imatge3),
      'informació mutua imatge 2 amb diferent:', info_mutua(mammarygland_2_imatge2,imatge_mostra_diferent)
      )




#30 de novemdre 2019 al 8 de desembre
import numpy as np
a=np.array([0.2,3])
b=np.array([1.9,9.2])
c=np.array([-6.2,5.9])
d=np.array([1,780])
e=np.array([8,4.3])
f=np.array([-19.1,3.21])

from skimage import data
coins = data.coins()

A=np.array([a,b,c,d])
A=np.maximum(A,0) # Si ens diu que ve de posicions negatives ho posam com si vengués de la corresponent coordenada 0
A[:,:,0]=np.minimum(A[:,:,0],coins.shape[0]-1) #la coordenada x no pot ser major que les files de l'imatge de referencia
A[:,:,1]=np.minimum(A[:,:,1],coins.shape[1]-1) #la coordenada y no pot ser major que les columnes de l'imatge dereferència

columna1=A[:,:,0]
columna2=A[:,:,1]
A21=np.floor(A)
A12=np.ceil(A)
A11=np.ones(A.shape)
A11[:,:,0]=np.floor(columna1)
A11[:,:,1]=np.ceil(columna2)
A22=np.ones(A.shape)
A22[:,:,0]=np.ceil(columna1)
A22[:,:,1]=np.floor(columna2)

A11=imatge_vec(A11,2).astype(int)
A12=imatge_vec(A12,2).astype(int)
A21=imatge_vec(A21,2).astype(int)
A22=imatge_vec(A22,2).astype(int)


#ara si d'una matriu, per exemple la imatge coins, en volem els valors que estan en les posicions A11 feim:



#SI TOT FOSSIN NO ENTERS FUNCIONA:
f11 = (1-(-A11[:,0]+A[:,0])*(A11[:,1]-A[:,1]))/3
f12 = (1-(A12[:,0]-A[:,0])*(A12[:,1]-A[:,1]))/3
f21 = (1-(-A21[:,0]+A[:,0])*(-A21[:,1]+A[:,1]))/3
f22 = (1-(A22[:,0]-A[:,0])*(-A22[:,1]+A[:,1]))/3

#np.where(A11[:,1]==A[:,1])#posicions on les columnes són enters
#np.where(A11[:,0]==A[:,0])#posicions on les files són enters.
#al cas que no tots son enters hem de canviar determinades posicions de les fij

#canviam les fij de les posicions on les COLUMNES són enters:
f11[np.where(A11[:,1]==A[:,1])] = (1-(-A11[:,0]+A[:,0]))[np.where(A11[:,1]==A[:,1])]
f12[np.where(A11[:,1]==A[:,1])] = (1-(A12[:,0]-A[:,0]))[np.where(A11[:,1]==A[:,1])]
f21[np.where(A11[:,1]==A[:,1])] = 0
f22[np.where(A11[:,1]==A[:,1])] = 0


#canviam les fij de les posicions on les FILES són enters:
f11[np.where(A11[:,0]==A[:,0])] = (1-(A11[:,1]-A[:,1]))[np.where(A11[:,0]==A[:,0])]
f12[np.where(A11[:,0]==A[:,0])] = 0
f21[np.where(A11[:,0]==A[:,0])] = (1-(-A21[:,1]+A[:,1]))[np.where(A11[:,0]==A[:,0])]
f22[np.where(A11[:,0]==A[:,0])] = 0

#si tant files com columnes són enters:
f11[np.where(np.sum(A11==A,axis=1)==2)] = 1
f12[np.where(np.sum(A11==A,axis=1)==2)] = 0
f21[np.where(np.sum(A11==A,axis=1)==2)] = 0
f22[np.where(np.sum(A11==A,axis=1)==2)] = 0

B=(f11*coins[A11[:,0],A11[:,1]]+f12*coins[A12[:,0],A12[:,1]]+f21*coins[A21[:,0],A21[:,1]]+f22*coins[A22[:,0],A22[:,1]])


#gener

'''
class BaseTransform:
    def find_best_transform(self, reference_image, input_image):
        raise NotImplementedError
    def apply_transform(self, input_image):
        raise NotImplementedError
    def visualize_transform(self):
        return None
        
class Rescala(BaseTransform):
    def __init__(self):
        self.dim_imatge = None
    def find_best_transform(self, reference_image, input_image):
        return reference_image.shape
    def apply_transform(self, input_image,dim):
        return resize(input_image,dim)
        
transformada = Rescala()
dim =transformada.find_best_transform(imatge1, imatge2)
imatge2rescalada= transformada.apply_transform(imatge2,dim)       
'''