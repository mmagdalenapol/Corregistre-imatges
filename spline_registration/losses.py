import numpy as np

def always_return_zero(reference_image, transformed_image):

    return 0

def SSD(reference_image, transformed_image):

    N = reference_image.shape[0]*reference_image.shape[1]
    a = reference_image-transformed_image

    SSD = np.sum(a * a) / N

    return SSD

def info_mutua(reference_image, transformed_image,n):
    #x és la reference_imatge, y la transformed_image
    #pxy distribucio de probabilitat conjunta
    #px i py distribucions marginals (la d'x s'obté sumant per files i la de y per columnes)


    from spline_registration.utils import descomposar
    imatge1d = descomposar(reference_image, n)
    imatge2d = descomposar(transformed_image, n)

    files = imatge1d.shape[0]
    columnes = imatge1d.shape[1]
    imatge1d = imatge1d.ravel()
    imatge2d = imatge2d.ravel()

    imatge1d = imatge1d.reshape(files * columnes, 3)
    imatge2d = imatge2d.reshape(files * columnes, 3)
    IMATGE1 = imatge1d[:, 0] * n + imatge1d[:, 1] + n * n * imatge1d[:, 2]
    IMATGE2 = imatge2d[:, 0] * n + imatge2d[:, 1] + n * n * imatge2d[:, 2]
    histograma = np.histogram2d(IMATGE1, IMATGE2, bins=n)

    pxy = histograma[0]/np.sum(histograma[0])
    px = pxy.sum(axis=1)#sumes els elements de la mateixa fila obtenim un array
    py = pxy.sum(axis=0)#sumes els elements de la mateixa columna

    #els pxy que siguin 0 no les tenim en compte ja que no aporten res
    # a la informació mutua i el log de 0 no està definit

    info_mutua = 0
    for i in range(0, pxy.shape[0]):
        for j in range(0, pxy.shape[1]):
            if pxy[i, j] != 0:
                info_mutua = info_mutua + pxy[i, j] * np.log(pxy[i, j] / (px[i] * py[j]))

    return info_mutua

#m'he basat amb https://matthew-brett.github.io/teaching/mutual_information.html per la informacio mutua
#com major és el nombre que ens torna menor és l'error entre les imatges
