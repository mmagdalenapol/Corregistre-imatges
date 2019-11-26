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
    contingut_carpeta_inici = os.listdir(carpeta_inici)#llista de les carpetes i arxius que hi ha dins la nostra carpeta d'inici
    direccions = {} #cream un diccionari inicialment buit

    subcarpetes_carpeta_inici = [path for path in contingut_carpeta_inici if mirar_carpeta(carpeta_inici + '/' + path)] #Guardam les que son carpetes
    for subcarpeta in subcarpetes_carpeta_inici:
        l = []
        direccio =carpeta_inici + '/' + subcarpeta
        j = -1

        while mirar_carpeta(direccio) == True:# Si és una carpeta haurem de seguir cercant dins la carpeta i amb aquesta comanda ho feim
            x=os.listdir(direccio)#llegeix sa carpeta de manera rara no entenc es criteri que segueix
            original=direccio
            for element in x:
                if mirar_carpeta(direccio+'/'+element) == False:  # així guardam tan sols les direccions de les imatges i no les de subcarpetes.
                    l.append(original+'/'+element)
                    j = j + 1
                    direccio = l[j]
                else:
                    direccio = direccio+'/'+element

        direccions[str(subcarpeta)] = l #al diccionari afegim una posició tal que tengui per nom el mateix nom que la subcarpeta i contengui
        # l'array amb les direccions de totes les imatges d'aquesta carpeta.


    for nom_carpeta, llista_imatges in direccions.items():
        pass

    for nom_carpeta in direccions.keys():
        pass

    for llista_imatges in direccions.values():
        pass


    return direccions


   #ara per cada subcarpeta trobada hauré de mirar dedins i així fins que arribi a que ja no es una carpeta que per tant serà una imatge i
    #guardaré la seva direcció de manera que després hi pugui accedir

def cerca_imatge_anhir(llista_de_dades, nomcarpeta, nomimatge):

    carpeta = llista_de_dades[nomcarpeta]
    #així com llista de dades es un diccionari podem cercar pel nom d'un element i ens torna el que hi ha a n'aquella posicio

    for imatge in carpeta:
        if imatge.find(nomimatge)>=0:
            #print('la imatge ' , nomimatge , ' de la carpeta ' , nomcarpeta , ' es troba a la direcció: ' ,  imatge )
            return imatge


    return {' no hi ha cap imatge amb el nom ' + nomimatge + ' a la carpeta ' + nomcarpeta }

    #aquest for tenc un exemple de com ficar a un yield text i el valor d'alguna variable.
    #for i in range(0,5):
    #      yield ('es troba a la secció {i}'.format(i=i))




#os.listdir(nom directori)#contingut de sa carpeta
#filtrar les que son carpetes os.path.isdir(path des fitxer que vols saber si es fitxer o carpeta) torna vertader si es carpeta fals si no ho és
