#26 novembre 2019

from spline_registration.losses import info_mutua
from spline_registration.databases import anhir
from spline_registration.databases import cerca_imatge_anhir
from skimage.io import imread
from spline_registration.transform_models import Rescala

n=5
imatge1=imread(cerca_imatge_anhir (anhir(),'mammary-gland_2', 's2_61-HE_A4926-4L.jpg'))
imatge2=imread(cerca_imatge_anhir (anhir(),'mammary-gland_2', 's2_62-ER_A4926-4L.jpg'))
imatge3=imread(cerca_imatge_anhir (anhir(),'mammary-gland_2', 's2_63-HE_A4926-4L'))
imatge2= Rescala.find_best_transform(Rescala(),imatge1,imatge2)
imatge3= Rescala.find_best_transform(Rescala(),imatge1,imatge3)

print('informació mutua imatge 1 amb 2:',info_mutua(imatge1, imatge2,n),
      'informació mutua imatge 1 amb 3:',info_mutua(imatge1, imatge3,n),
      'informació mutua imatge 2 amb 3:',info_mutua(imatge2, imatge3,n))



#veim que la informació mutua es menor quan es tracta de imatges de mostres diferents.
imatge_mostra_diferent= imread(cerca_imatge_anhir (anhir(),'lung-lesion_1', '29-041-Izd2-w35-Cc10-5-les1'))
imatge_mostra_diferent= Rescala.find_best_transform(Rescala(),imatge1,imatge_mostra_diferent)
print('informació mutua imatge 1 amb diferent:', info_mutua(imatge1, imatge_mostra_diferent,n),
      'informació mutua imatge diferent amb 3:', info_mutua(imatge_mostra_diferent, imatge3,n),
      'informació mutua imatge 2 amb diferent:', info_mutua(imatge2,imatge_mostra_diferent,n)
      )

print('una imatge amb ella mateixa te una info mutua molt més gran'info_mutua(imatge1, imatge1,n))