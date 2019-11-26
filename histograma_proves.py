#funcio per imatges en dues dimensions RGB

#volem descomposar els colors d'una imatge agrupant els R,G,B en n invervals (normalment 5 o 10, així hi hauria 125 0 1000 grups diferents de colors.)

#introduim per paràmetre la imatge i n
import numpy as np


def descomposar(imatge,n):
    files = imatge.shape[0]
    columnes = imatge.shape[1]
    rangcolors = np.linspace(imatge.min(), imatge.max(), n)
    classe_imatge = np.ones((imatge.shape)) #aqui guardam cada pixel a quin grup pertany
    for fila in range(0, files):
        for columna in range(0, columnes):
            color = imatge[fila,columna] #cada pixel està format per un array amb tres nombres R,G,B
            j=0 #contador de l'index R G B
            for col in color:
                a = rangcolors - col
                i=0
                while i in range(0, len(a)):
                    if a[i] >= 0: #ens interessa que la diferència sigui positiva i aixó pertany a aquest grup
                        if i == 0:
                            classe_imatge[fila, columna][j] = rangcolors[i]
                            i = len(a)
                        else:
                            if a[i - 1] < 0:
                                classe_imatge[fila, columna][j] = rangcolors[i - 1]
                                i = len(a)

                    else:
                        i=i+1
                j=j+1
    return classe_imatge



##ara la funció que fa bàsicament el mateix però molt més senzill:

def descomposar (imatge,n):

    imatge = (imatge - imatge.min())/(imatge.max()-imatge.min()) #aixi tots els valors de la imatge van entre 0 i 1

    imatge = np.floor_divide(imatge, 1/n)  #ara cada valor de la imatge és la seva classe.

    return imatge





import numpy as np
from matplotlib import pyplot as plt
from skimage import data
cat = data.chelsea()
imatge=descomposar(cat,5)



##altre opció:
import numpy as np
from spline_registration.databases import anhir
from spline_registration.databases import cerca_imatge_anhir
from skimage.io import imread
from spline_registration.transform_models import Rescala

imatge1 = imread(cerca_imatge_anhir (anhir(),'mammary-gland_2', 's2_64-PR_A4926-4L'))
imatge2 = imread(cerca_imatge_anhir (anhir(),'mammary-gland_2', 's2_62-ER_A4926-4L'))
imatge2 = Rescala.find_best_transform(Rescala(),imatge1,imatge2)#rescalam la imatge 2 per tal que sigui de la mateixa dimensio que la 1

imatge_mostra_diferent = imread(cerca_imatge_anhir (anhir(),'lung-lesion_1', '29-041-Izd2-w35-Cc10-5-les1'))
imatge_mostra_diferent = Rescala.find_best_transform(Rescala(),imatge1,imatge_mostra_diferent)


from spline_registration.losses import info_mutua
print('la informacio mutua de 2 imatges de la mateixa mostra:',info_mutua(imatge2, imatge1,5))

print('la informacio mutua de 2 imatges de mostres diferents:',info_mutua(imatge1, imatge_mostra_diferent,5))

print('la informacio mutua de 2 imatges de la mateixa mostra:',info_mutua(imatge1, imatge1,5))