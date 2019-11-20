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
imatge2 = Rescala.find_best_transform(Rescala(),imatge1,imatge2)#rescalam la imatge 2 per tal que sigui de la mateixa dimensio que la 1

visualize_side_by_side(imatge1,imatge2, 'Esquerra: s2_64-PR_A4926-4L.jpg ; Dreta: s2_62-ER_A4926-4L.jpg ')
print('L´error SSD entre les imatges de la mateixa mostra és:',SSD(imatge1, imatge2))
print('L´informació mútua entre les imatges de la mateixa mostra és:',info_mutua(imatge1, imatge2))
# com podem veure les 2 imatges són semblants però de colors diferents i per aquesta raó ja ens dóna un error molt gran.
# hem de trobar una mesura d'error millor.

imatge_mostra_diferent = imread(cerca_imatge_anhir (anhir(),'lung-lesion_1', '29-041-Izd2-w35-Cc10-5-les1'))
imatge_mostra_diferent = Rescala.find_best_transform(Rescala(),imatge1,imatge_mostra_diferent)


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





n=5
imatge1d=descomposar(imatge1,n)
imatge2d=descomposar(imatge2,n)

files=imatge1d.shape[0]
columnes=imatge1d.shape[1]
imatge1d=imatge1d.ravel()
imatge2d=imatge2d.ravel()

imatge1d=imatge1d.reshape(files*columnes,3)
imatge2d=imatge2d.reshape(files*columnes,3)

IMATGE1=imatge1d[:,0]*n+imatge1d[:,1]+n*n*imatge1d[:,2]
IMATGE2=imatge2d[:,0]*n+imatge2d[:,1]+n*n*imatge2d[:,2]
histograma = np.histogram2d(IMATGE1, IMATGE2,bins=n)

