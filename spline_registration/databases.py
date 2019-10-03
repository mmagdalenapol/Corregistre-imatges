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

#torna vertader si es carpeta
def mirar_carpeta(pathcarpeta):
    return os.path.isdir(pathcarpeta)


def anhir():
    carpeta_inici = base_path('databases/anhir/dataset_medium')
    contingut_carpeta_inici=os.listdir(carpeta_inici)#llista de les carpetes i arxius que hi ha dins la nostra carpeta d'inici
    direccions=[]
    i=-1
    for subcarpeta in contingut_carpeta_inici:
        i=i+1
        l = []
        direccio0=carpeta_inici + '/' + subcarpeta
        j=-1

        while mirar_carpeta(direccio0) == True:# Si és una carpeta haurem de seguir cercant dins la carpeta i amb aquesta comanda ho feim
            x=os.listdir(direccio0)#llegeix sa carpeta de manera rara no entenc es criteri que segueix
            for element in x:
                direccio = direccio0
                if mirar_carpeta(direccio+'/'+element) == False:  # així guardam tan sols les direccions de les imatges i no les de subcarpetes.
                    l.append(direccio+'/'+element)
                    j = j + 1
                    direccio0 = l[j]
                else:
                    direccio = direccio+'/'+element


        direccions.append(l)

    return direccions


   #ara per cada subcarpeta trobada hauré de mirar dedins i així fins que arribi a que ja no es una carpeta que per tant serà una imatge i
    #guardaré la seva direcció de manera que després hi pugui accedir

def cerca_imatge_anhir(llista_de_dades, nomcarpeta, nomimatge):

    carpeta_inici = base_path('databases/anhir/dataset_medium')
    contingut_carpeta_inici = os.listdir(carpeta_inici)
    carpeta = llista_de_dades[contingut_carpeta_inici.index(nomcarpeta)]#com llegeix les carpetes en un ordre extrany així localitzam les fotos d'una determinada carpeta
    for imatge in carpeta:
        if imatge.find(nomimatge)>=0:
            print('la imatge' , nomimatge , ' de la carpeta ' , nomcarpeta , 'es troba a la direcció:' ,  imatge )
            return imatge


    return {'no hi ha cap imatge amb el nom ' + nomimatge + ' a la carpeta ' + nomcarpeta }

    #aquest for tenc un exemple de com ficar a un yield text i el valor d'alguna variable.
    #for i in range(0,5):
    #      yield ('es troba a la secció {i}'.format(i=i))




#os.listdir(nom directori)#contingut de sa carpeta
#filtrar les que son carpetes os.path.isdir(path des fitxer que vols saber si es fitxer o carpeta) torna vertader si es carpeta fals si no ho és
