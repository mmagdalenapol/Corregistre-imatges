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

#si és carpeta torna vertader
def mirar_carpeta(pathcarpeta):
    return os.path.isdir(pathcarpeta)
'''
Quan volem carregar les imatges el primer que hem de fer és arribar a elles.
Per això he creat la mirar_carpeta(direccio del directori).
hi ha una funció de la llibreria os que fa precisament això. 
Aquesta funció és os.path.isdir(pathcarpeta).
Que el que fa es retorna vertader si és carpeta i fals sinó.
'''


def anhir():
    carpeta_inici = base_path('databases/anhir/dataset_medium')
    '''
       Feim feina amb les imatges de anhir.
       Per tant el primer que feim es indicar la carpeta on hi haurà totes les subcarpetes que anirem cercant.
       Al nostre cas tot el que va abans de database no ho hem de posar perquè el nostre directori de feina es 
       precisament spline_registration.
       
    '''
    contingut_carpeta_inici = os.listdir(carpeta_inici)
    '''
    Ara guardam a contingut_carpeta_inici una llista de les 
    carpetes i arxius que hi ha dins la nostra carpeta d'inici
    '''

    direccions = {} #cream un diccionari inicialment buit

    subcarpetes_carpeta_inici = [path for path in contingut_carpeta_inici if mirar_carpeta(carpeta_inici + '/' + path)]
    #Guardam els path de les que son carpetes

    for subcarpeta in subcarpetes_carpeta_inici:
        l = []
        direccio =carpeta_inici + '/' + subcarpeta
        j = -1
        '''
        el vector l es on guardarem els path de les imatges.
        direccio inicialitza a quina subcarpeta ens trobam en cada iteracio del for
        la variable j serveix per indicar la posició del vector l on va el path 
        '''
        while mirar_carpeta(direccio) == True:
            # Si és una carpeta haurem de seguir cercant dins la carpeta i amb aquesta comanda ho feim
            x=os.listdir(direccio)#llegeix sa carpeta de manera rara no entenc es criteri que segueix
            original=direccio
            for element in x:
                if mirar_carpeta(direccio+'/'+element) == False:
                    # així guardam tan sols les direccions de les imatges i no les de subcarpetes.
                    l.append(original+'/'+element)
                    j = j + 1
                    direccio = l[j]
                else:
                    direccio = direccio+'/'+element

        direccions[str(subcarpeta)] = l
        '''
        al diccionari afegim una posició que tengui per nom el mateix nom que la subcarpeta 
        i contengui l'array amb les direccions de totes les imatges d'aquesta carpeta.
        '''

    '''
    la funcio anhir ens torna el diccionari direccions que te per keys els noms de les subcarpetes i per values els paths de les imatges dins aquestes
    '''

    for nom_carpeta, llista_imatges in direccions.items():
        pass

    for nom_carpeta in direccions.keys():
        pass

    for llista_imatges in direccions.values():
        pass


    return direccions


def cerca_imatge_anhir(llista_de_dades, nomcarpeta, nomimatge):

    carpeta = llista_de_dades[nomcarpeta]
    '''
    com llista de dades es un diccionari podem cercar pel nom d'un element 
    i ens torna el que hi ha en aquella posició.
    '''

    for imatge in carpeta:
        if imatge.find(nomimatge)>=0:
            'donada una cadena de caràcters find ens cerca si hi ha una combinacio de lletres dedins.' \
            'en casafirmatiu dona un nombre positiu i vol dir que hem trobat la imatge desitjada.'
            #print('la imatge ' , nomimatge , ' de la carpeta ' , nomcarpeta , ' es troba a la direcció: ' ,  imatge )
            return imatge


    return {' no hi ha cap imatge amb el nom ' + nomimatge + ' a la carpeta ' + nomcarpeta }

