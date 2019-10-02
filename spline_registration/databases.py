from skimage.io import imread
from skimage import io
from spline_registration.utils import get_databases_path
from spline_registration.utils import base_path
import os


def anhir_test():
    path_reference = f'{get_databases_path()}/anhir_test/S2.jpg'
    path_input = f'{get_databases_path()}/anhir_test/HE.jpg'
    yield {
        'reference': imread(path_reference),
        'input': imread(path_input),
        'name': 'Test Image 01',
        'expert_information': None,
    }

def anhir():
    carpeta_inici = base_path('databases/anhir/dataset_medium')
    contingut_carpeta_inici=os.listdir(carpeta_inici)#llista de les carpetes i arxius que hi ha dins la nostra carpeta d'inici
    for subcarpeta in contingut_carpeta_inici:
        if os.path.isdir(carpeta_inici+'/'+subcarpeta) == True:# Si és una carpeta haurem de seguir cercant dins la carpeta i aquesta comanda

   #ara per cada subcarpeta trobada hauré de mirar dedins i així fins que arribi a que ja no es una carpeta que per tant serà una imatge i
    #guardaré la seva direcció de manera que després hi pugui accedir



    #aquest for tenc un exemple de com ficar a un yield text i el valor d'alguna variable.
    #for i in range(0,5):
    #      yield ('es troba a la secció {i}'.format(i=i))




#os.listdir(nom directori)#contingut de sa carpeta
#filtrar les que son carpetes os.path.isdir(path des fitxer que vols saber si es fitxer o carpeta) torna vertader si es carpeta fals si no ho és
