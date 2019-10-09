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
from spline_registration.transform_models import Rescala

imatge1 = imread(cerca_imatge_anhir (anhir(),'COAD_05', 'S6'))
imatge2 = imread(cerca_imatge_anhir (anhir(),'COAD_05', 'S7'))
imatge2 = Rescala.find_best_transform(Rescala(),imatge1,imatge2)#rescalam la imatge 2 per tal que sigui de la mateixa dimensio que la 1


visualize_side_by_side(imatge1,imatge2, 'Esquerra: S6.jpg ; Dreta: S7.jpg ')
print('L´error entre les imatges és:',SSD(imatge1, imatge2))
# com podem veure les 2 imatges són semblants però de colors diferents i per aquesta raó ja ens dóna un error molt gran.
# hem de trobar una mesura d'error millor.

