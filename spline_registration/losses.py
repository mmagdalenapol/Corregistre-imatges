import numpy as np
from spline_registration.utils import descomposar


def SSD(reference_image, transformed_image):

    N = reference_image.shape[0] * reference_image.shape[1]
    a = reference_image - transformed_image
    '''
    N és el nombre de pixels de la imatge de referència
    suposam ambdues imatges de la mateixa dimensió ja quan les introduim.
    a són els errors en els valors de les imatges
    '''
    SSD = np.sum(a * a) / N

    return SSD


def RMSE(reference_image, transformed_image):

    N = reference_image.shape[0] * reference_image.shape[1]
    dif = reference_image - transformed_image
    RMSE = np.sqrt(np.sum(dif * dif)) / N
    return RMSE


def info_mutua(reference_image, transformed_image, n):
    '''
    n ens dona en quants de grups dividim cada color

    x és la reference_imatge, y la transformed_image
    pxy distribucio de probabilitat conjunta
    px i py distribucions marginals (la d'x s'obté sumant per files i la de y per columnes).
    '''

    imatge1 = descomposar(reference_image, n)
    imatge2 = descomposar(transformed_image, n)

    '''
    descomposant reference_image i transformed_image obtenim dos vectors d'una fila i 
    reference_image.shape[0]*reference_image.shape[1] columnes
    on cada element és la classe del color que hi havia abans. 
    '''

    histograma = np.histogram2d(imatge1, imatge2, bins=(n**3))

    '''
    ara feim un histograma que histograma[0] conté els pics que la imatge1 val un valor mentre que a la imatge2 un altre
    per totes les combinacions possibles. Per tant la distibició de probabilitat conjunta és histograma[0] normalitzat.
    '''

    pxy = histograma[0]/np.sum(histograma[0])
    px = pxy.sum(axis=1)  # sumes els elements de la mateixa fila obtenim un array
    py = pxy.sum(axis=0)  # sumes els elements de la mateixa columna

    # els pxy que siguin 0 no les tenim en compte ja que no aporten res
    # a la informació mutua i el log de 0 no està definit

    pxy = histograma[0] / np.sum(histograma[0])

    PX = np.transpose(np.tile(px, (n ** 3, 1)))
    PY = np.tile(py, (n ** 3, 1))
    den = PX * PY
    den2 = np.where(den == 0, 1, den)
    num = pxy
    log = np.log(np.where(den*num == 0, 1, num / den2))
    mutual_info = np.sum(pxy*log)

    return (pxy*log).ravel()
    # return mutual_info