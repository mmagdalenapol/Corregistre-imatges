import numpy as np
from time import time

def always_return_zero(reference_image, transformed_image):

    return 0

def SSD(reference_image, transformed_image):

    N = reference_image.shape[0]*reference_image.shape[1]
    a = reference_image-transformed_image
    '''
    N és el nombre de pixels de la imatge de referència
    suposam ambdues imatges de la mateixa dimensió ja quan les introduim.
    a són els errors en els valors de les imatges
    '''
    SSD = np.sum(a * a) / N

    return SSD

def info_mutua(reference_image, transformed_image,n):
    '''
    n ens dona en quants de grups dividim cada color

    x és la reference_imatge, y la transformed_image
    pxy distribucio de probabilitat conjunta
    px i py distribucions marginals (la d'x s'obté sumant per files i la de y per columnes)
    '''

    from spline_registration.utils import descomposar
    imatge1 = descomposar(reference_image, n)
    imatge2 = descomposar(transformed_image, n)
    '''
    descomposant reference_image i transformed_image obtenim dos vectors d'una fila i reference_image.shape[0]*reference_image.shape[1] columnes
    on cada element és la classe del color que hi havia abans. 
    '''

    histograma = np.histogram2d(imatge1, imatge2,bins=(n**3))
    '''
    ara feim un histograma que histograma[0] conté els pics que la imatge1 val un valor mentre que a la imatge2 un altre
    per totes les combinacions possibles. Per tant la distibició de probabilitat conjunta és histograma[0] normalitzat.
    '''

    pxy = histograma[0]/np.sum(histograma[0])
    px = pxy.sum(axis=1)#sumes els elements de la mateixa fila obtenim un array
    py = pxy.sum(axis=0)#sumes els elements de la mateixa columna

    #els pxy que siguin 0 no les tenim en compte ja que no aporten res
    # a la informació mutua i el log de 0 no està definit

    pxy = histograma[0] / np.sum(histograma[0])

    temps_matrius_inicial = time()

    PX = np.transpose(np.tile(px, (n ** 3, 1)))
    PY = np.tile(py, (n ** 3, 1))
    den = PX * PY
    den2 = np.where(den==0,1,den)
    num = pxy
    log = np.log(np.where(den*num==0,1,num/den2))
    mutual_info = np.sum(pxy*log)

    temps_matrius_final = time ()
    temps_execucio_1 = temps_matrius_final - temps_matrius_inicial
    '''
    el den pot ser 0 i per tant això duu problemes a l'hora de dividir. Per això definim den2 que és igual a den per tot 
    menys on den val 0 que posam un 1, realment mai emplearem aquest 1 però així evitam el troblema de dividir entre 0.
    
    Ara allà on o bé el numerador o denominador inicial (és a dir den) valen 0 posam un 1 ja que np.log(1)=0. i a la 
    resta de posicions calculam el logaritme de num/den2 (que en aquestes posicions és el mateix que num/den però fent 
    això hi havia problemes ja que l'ordenador feia divisions entre 0 encara que no les emprés).
     
    Amb això ja tenim que la informació mútua és la suma de pxy*log.
    
    Inicialment ho havia fet amb aquest bucle (que també funciona però tarda més):
    
    temps_bucle_inicial = time()
    info_mutua = 0
    for i in range(0, pxy.shape[0]):
        for j in range(0, pxy.shape[1]):
            if pxy[i, j] != 0:
                info_mutua = info_mutua + pxy[i, j] * np.log(pxy[i, j] / (px[i] * py[j]))

    temps_bucle_final = time()

    temps_execucio_2 = temps_bucle_final -temps_bucle_inicial
    
    print(temps_execucio_1<temps_execucio_2)
    amb aquesta simple comprovació que ens dóna True veim que efectivament és millor fer-ho amb les matrius que amb el bucle
    
    '''
    return mutual_info



'''
m'he basat amb https://matthew-brett.github.io/teaching/mutual_information.html per la informacio mutua del bucle encara
que no és amb la que faig feina al final.

com major és el nombre que ens torna menor és l'error entre les imatges.
'''

