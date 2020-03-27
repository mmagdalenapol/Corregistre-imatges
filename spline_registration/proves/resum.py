#3 febrer 2020
'''
El nostre objectiu és fer un corregistre elàstic entre dues imatges.
Per poder fer-ho el primer que feim és carregar les nostres imatges.
Així el primer que vaig fer va ser descobrir com donada una carpeta amb subcarpetes anar guardant les direccions de totes les imatges.
I així poder accedir-hi ràpid quan volguem fer feina amb una determinada imatge.
Això ho vaig fer al fitxer databases.

Si volem fer feina amb unes dades diferents a anhir el que hem de fer és
guardar la carpeta com una subcarpeta de databases. Ara tenim la funció anhir()
del fitxer databases que bàsicament ens retorna un diccionari amb keys les carpetes i valors els paths de les respectives imatges.

Amb un altre conjunt d'imatges podriem crear fàcilment una funció semblant que faci el mateix.
'''

from spline_registration.databases import anhir, cerca_imatge_anhir
data_anhir = anhir()
print(data_anhir.keys())

'''
Ara ja hem vist que tenim un diccionari amb keys els nom de les subcarpetes i
els valors els paths de cada una de les imatges que hi ha dins cada subcarpeta.
'''

#5 de febrer
'''
Un pic tenim les direccions el següent seria guardar una imatge sabent la seva direcció.

-Obtenir direcció a partir del nom i carpeta:
    Per això vaig fer cerca_imatge_anhir(llista_de_dades, nomcarpeta, nomimatge)
    al nostre exemple:
    llista_de_dades = data_anhir i si per exemple volem la imatge 's2_64-PR_A4926-4L' de la subcarepta 'mammary-gland_2' 
    fent cerca_imatge_anhir(data_anhir, 'mammary-gland_2', 's2_64-PR_A4926-4L') obtenim el path d'aquesta imatge 
    (en cas d'existir, si no existís tornaria un missatge d'error)
    
-carregar la imatge 

    aquí ens és de gran utilitat el paquet skimage de processament d'imatges de python.
    utilitzam el modul io d'aquest paquet que ens permet llegir
    si la volem visualizar podem utilitzar del paquet matplotlib el modul pyplot
'''

path_imatge1 = cerca_imatge_anhir (anhir(),'mammary-gland_2', 's2_64-PR_A4926-4L')

from skimage.io import imread
import matplotlib.pyplot as plt

imatge1= imread(path_imatge1)


#plt.imshow(imatge1)
#plt.title("imatge1 ('s2_64-PR_A4926-4L' de la carpeta 'mammary-gland_2')")
#plt.show()
#plt.close()

'''
Com voldrem fer el corregistre entre dues imatges ens vendrà bé poder comparar les 2 imatges una devora l'altre.
Per això és que si ara a part de la imatge1 carregam una altre imatge, imatge 2 (de la mateixa mostra per començar), 
les podem representar una devora l'altre amb la funció visualize_side_by_side(image_left, image_right, title=None) de utils. 
'''

imatge2 = imread(cerca_imatge_anhir (anhir(),'mammary-gland_2', 's2_62-ER_A4926-4L'))

from spline_registration.utils import visualize_side_by_side

#visualize_side_by_side(imatge1, imatge2, title=None)



'''
11 de febrer

Arribat a n'aquest punt ja toca començar a fer feina amb les imatges.
Per començam feim una mesura molt simple de perdua: SSD.
Aquí em vaig trobar amb un primer problema les imatges no tenien la mateixa dimensió. Per això a transform_models
vaig crear la classe Rescala.

Rescala ens permet obtenir una imatge de la mateixa dimensió que la de referència.  
'''
imatge_mostra_diferent = imread(cerca_imatge_anhir (anhir(),'lung-lesion_1', '29-041-Izd2-w35-Cc10-5-les1'))

print('\n dimensió imatge1:',imatge1.shape   ,'\n dimensió imatge_mostra_diferent',imatge_mostra_diferent.shape,'\n veim que les dimensions originalment són diferents') #veim que les dimensions originalment són diferents.

from spline_registration.transform_models import Rescala

transformada=Rescala()
'transformada és un objecte de la classe rescala'

from skimage.transform import resize

imatge_mostra_diferent = transformada.apply_transform(imatge1,imatge_mostra_diferent)#imatge_mostra_diferent_rescalada
print('\n dimensió imatge1:',imatge1.shape,'\n dimensió imatge_mostra_diferent',imatge_mostra_diferent.shape, '\n veim que ara les dimensions són les mateixes')

'''
Ara que ja tenim imatges de la mateixa dimensió fàcilment (donada la imatge de referència i la que volem canviar) podem calcular la mesura d'error ssd
'''
from spline_registration.losses import SSD
print('\nL´error SSD entre les imatges de la mateixa mostra és:',SSD(imatge1, imatge2),
      '\nL´error SSD entre les imatges de mostra diferent és:',SSD(imatge1, imatge_mostra_diferent),
      '\n veim que de la mateixa mostra és menor però és una mesura d´error no gaire bona')

'''
24 de març
Per tant calculam una mesura d'error millor : info_mutua.
La informació mutua mesura la informació que dues imatges A i B comparteixen.
Si són indepentdents una no dóna informació de l'altre i per tant la informació mútua és 0. 
'''

from spline_registration.losses import info_mutua
print('\nL´informació mútua entre les imatges de la mateixa mostra és:',info_mutua(imatge1, imatge2,5),
      '\nL´informació mútua entre les imatges de mostra diferent és:',info_mutua(imatge1, imatge_mostra_diferent,5),
      '\n veim que de la mateixa mostra té més informació mútua que de mostres diferents')


'''  
26 de març

Ara que ja tenim una mesura d'error per poder comparar dues imatges un cop tenim el corregistre fet 
hem de començar a fer alguna cosa relacionada amb el corregistre.

Suposem que tenim la imatge corregistrada i l'original. De la corregistrada tenim uns certs punts (keypoints) dels quals
sabem les coordenades d'on provenen a la imatge original. El primer que he fet es trobar els pixels d'aquestes coordenades
ja que poden pasar diverses coses:
-les coordenades són decimals i per tant hem de saber a quin pixel fan referència. Per això feim una interpolació dels colors dels pixels més propers.
-les coordenades estàn fora de les dimensions de la imatge. 

a transforms_models hi ha la classe elastic transform que té la funció colors_transform on he intentat solucionar el problema dels colors amb una interpolació bilineal.
'''

