#funcio per imatges en dues dimensions RGB

#volem descomposar els colors d'una imatge agrupant els R,G,B en n invervals (normalment 5 o 10, així hi hauria 125 0 1000 grups diferents de colors.)

#introduim per paràmetre la imatge i n

def descomposar(imatge,n):
    files = imatge.shape[0]
    columnes = imatge.shape[1]
    rangcolors = np.linspace(imatge[:, :].min(), imatge[:, :].max(), n)
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

