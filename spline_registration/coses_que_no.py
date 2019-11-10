#funcio per imatges en dues dimensions RGB no ho hem de fer així perquè tarda un muunt

#volem descomposar els colors d'una imatge agrupant els R,G,B en n invervals (normalment 5 o 10, així hi hauria 125 0 1000 grups diferents de colors.)

#introduim per paràmetre la imatge i n
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

def info_mutua(reference_image, transformed_image):
    #x és la reference_imatge, y la transformed_image
    #pxy distribucio de probabilitat conjunta
    #px i py distribucions marginals (la d'x s'obté sumant per files i la de y per columnes)

    histograma = np.histogram2d(reference_image.ravel(), transformed_image.ravel())
    #plt.imshow(histograma[0],origin='lower') si volguessim dibuixar l'histograma
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
