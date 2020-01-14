import numpy as np

def always_return_zero(reference_image, transformed_image):

    return 0

def SSD(reference_image, transformed_image):

    N = reference_image.shape[0]*reference_image.shape[1]
    a = reference_image-transformed_image
    '''
    N és el nombre de pixels de la imatge de referència
    suposam ambdues imatges de la mateixa dimensio ja quan les introduim
    a són els errors en els valors de les imatges
    '''
    SSD = np.sum(a * a) / N

    return SSD

def info_mutua(reference_image, transformed_image,n):
#n ens dona en quants de grups dividim cada color

    #x és la reference_imatge, y la transformed_image
    #pxy distribucio de probabilitat conjunta
    #px i py distribucions marginals (la d'x s'obté sumant per files i la de y per columnes)


    from spline_registration.utils import descomposar
    imatge1 = descomposar(reference_image, n)
    imatge2 = descomposar(transformed_image, n)

    histograma = np.histogram2d(imatge1, imatge2,bins=(n**3))

    pxy = histograma[0]/np.sum(histograma[0])
    px = pxy.sum(axis=1)#sumes els elements de la mateixa fila obtenim un array
    py = pxy.sum(axis=0)#sumes els elements de la mateixa columna

    #els pxy que siguin 0 no les tenim en compte ja que no aporten res
    # a la informació mutua i el log de 0 no està definit

    '''
    pxy = histograma[0] / np.sum(histograma[0])
    logpxy=np.log(np.where(pxy==0,1,pxy))
    PX=np.transpose(np.tile(px,(n**3,1)))
    PY=np.tile(py,(n**3,1))
    num= pxy*logpxy
    den= PX*PY
    
    np.sum(np.where(num*den==0,0,num/den))
    
    '''

    info_mutua = 0
    for i in range(0, pxy.shape[0]):
        for j in range(0, pxy.shape[1]):
            if pxy[i, j] != 0:
                info_mutua = info_mutua + pxy[i, j] * np.log(pxy[i, j] / (px[i] * py[j]))

    return info_mutua




#m'he basat amb https://matthew-brett.github.io/teaching/mutual_information.html per la informacio mutua
#com major és el nombre que ens torna menor és l'error entre les imatges
