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

