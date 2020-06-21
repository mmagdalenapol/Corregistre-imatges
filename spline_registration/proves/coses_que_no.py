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




#sa meva intenció es contar el nombre de vegades que surt cada element .
valors = [imatge[0,0]]
nobs = [np.count_nonzero( np.sum( (np.where(imatge ==imatge[0,0], 1, 0)),axis=2) == 3 )]

for n in valors:

   if  np.any(np.floor_divide(imatge, valors[0])!=[0.,0.,0.]):
       valor= imatge[
           np.where(np.floor_divide(imatge, valors[0]) != [0., 0., 0.])[0][0],
           np.where(np.floor_divide(imatge, valors[0]) != [0., 0., 0.])[1][0],
           np.where(np.floor_divide(imatge, valors[0]) != [0., 0., 0.])[2][0]]

       valors.append( valor)
       nobs.append(np.count_nonzero( np.sum( (np.where(imatge ==valor, 1, 0)),axis=2) == 3 ))



valors = [imatge[0,0]]
imatge2 =np.sum( (np.where(imatge ==imatge[0,0], 1, 0)),axis=2) == 3
nobs=[np.count_nonzero(imatge2)]
indexx=np.where(imatge2 == True)[0]
indexy=np.where(imatge2 == True)[1]

for x in range (0, imatge.shape[0]):
    for y in range (0, imatge.shape[1]):
        if x in indexx and y not in indexy[np.where(indexx ==x)]:
            valor = imatge[x,y]
            imatge2 =np.sum( (np.where(imatge ==valor, 1, 0)),axis=2) == 3
            np.append(nobs,np.count_nonzero(imatge2))
            np.append(indexx,np.where(imatge2 == True)[0])
            np.append(indexy,np.where(imatge2 == True)[1])

        if x not in indexx:
            valor = imatge[x, y]
            imatge2 = np.sum((np.where(imatge == valor, 1, 0)), axis=2) == 3
            np.append(nobs, np.count_nonzero(imatge2))
            np.append(indexx, np.where(imatge2 == True)[0])
            np.append(indexy, np.where(imatge2 == True)[1])


    def colors_transform(self, reference_image, input_image):
        A = self.apply_transform(reference_image, input_image)

        '''
        A
        la transformació m'hauria de donar les coordenades d'on se suposa que ve cada pixel de la imatge corregistrada.
        Aquestes coordenades poden ser decimals i per això el valor del color depèn del color dels 4 pixels més propers.
        A encara és una matriu

        Abans de res feim uns quantes adaptacions inicials a A. 
        -No hi ha coordenades negatives per tant qualsevol element negatiu el passam a nes 0.
        -La coordenada x no pot ser major que les files de la imatge de referència.
        -La coordenada y no pot ser major que les columnnes de la imatge de referència.
        '''

        A = np.maximum(A, 0)
        A[:, :, 0] = np.minimum(A[:, :, 0], reference_image.shape[0] - 1)
        A[:, :, 1] = np.minimum(A[:, :, 1], reference_image.shape[1] - 1)

        columna1 = A[:, :, 0]
        columna2 = A[:, :, 1]

        '''
        A11 A12
        A21 A22

        per tant:
         A11: és la x floor i la y ceil      A12: és la x i la y ceil 
         A21: és la x i la y floor           A22: és la x ceil i la y floor
        '''

        A11 = np.ones(A.shape)
        A11[:, :, 0] = np.floor(columna1)
        A11[:, :, 1] = np.ceil(columna2)

        A12 = np.ceil(A)

        A21 = np.floor(A)

        A22 = np.ones(A.shape)
        A22[:, :, 0] = np.ceil(columna1)
        A22[:, :, 1] = np.floor(columna2)

        '''
        Ara el problema que tenim és que els nombres encara que són enters les tracta com reals i per tant ara passam 
        totes les matrius a vectors (perquè és més fàcil fer feina amn ells) i a més deim que A11,A12,A21 i A22 són de tipus enters  
        '''
        A = imatge_vec(A, 2)
        A11 = imatge_vec(A11, 2).astype(int)
        A12 = imatge_vec(A12, 2).astype(int)
        A21 = imatge_vec(A21, 2).astype(int)
        A22 = imatge_vec(A22, 2).astype(int)

        '''
        f11,f12,f21 i f22 són els coeficients per la interpolació bilineal. 
        és a dir color(x,y) = color(A21(x,y))*f21(x,y) + color(A22(x,y))*f22(x,y) + color(A11(x,y))*f11(x,y) + color(A12(x,y))*f12(x,y)

        així inicialment les possam tots 0 i les anam canviant segons si com sigui (x,y). 
        '''
        f11 = np.zeros(A.shape[0])  # només necessitam un nombre per posició
        f12 = np.zeros(A.shape[0])
        f21 = np.zeros(A.shape[0])
        f22 = np.zeros(A.shape[0])

        '''
        denx,deny,num21x,num21y,num11y,num22x ens serviràn per escriuré més fàcilment qui són f11,f12,f21 i f22 en cada cas
        '''
        denx = (A12[:, 0] - A21[:, 0])
        deny = (A12[:, 1] - A21[:, 1])

        num21x = A12[:, 0] - A[:, 0]
        num21y = A12[:, 1] - A[:, 1]
        num11y = A21[:, 1] - A[:, 1]
        num22x = A21[:, 0] - A[:, 0]

        '''
        inicialment totes són 0 i segons quina situació estam ho anirem canviant.

        Si la coordenada (x,y) ja eren els dos nombres enters llavors ens trobam al cas (x,y) = (np.ceil(x), np.ceil(y)) 
        i per tant si volem trobar els pixels que satisfan aquesta condició el que volem és trobar aquelles posicions tals que 
        A==A21. Això es equivalent als llocs on denx + deny == 0). 

        El cas en què amdbues coordenades són decimals és equivalent a dir denx+deny==2. 
        per tant en aquesta situació hem de canviar tots els valors.

        Cas x decimal i y entera equival a deny=0 i denx=1. Per tant en aquest cas els afectats són:  deny - denx = -1
        Cas x enter i y decimal equival a deny=1 i denx=0. Per tant equival a: deny - denx = 1
        '''
        f21[np.where(denx + deny == 0)] = 1

        den = np.where(denx + deny == 2, denx * deny, 1)  # per evitar problemes interns dividint entre 0
        f21[np.where(denx + deny == 2)] = ((num21x * num21y) / den)[np.where(denx + deny == 2)]
        f22[np.where(denx + deny == 2)] = ((num22x * num21y) / -den)[np.where(denx + deny == 2)]
        f11[np.where(denx + deny == 2)] = ((num21x * num11y) / -den)[np.where(denx + deny == 2)]
        f12[np.where(denx + deny == 2)] = ((num22x * num11y) / den)[np.where(denx + deny == 2)]

        denxmodificat = np.where(deny - denx == -1, denx, 1)  # per evitar problemes interns dividint entre 0
        f21[np.where(deny - denx == -1)] = (num21x / denxmodificat)[np.where(deny - denx == -1)]
        f22[np.where(deny - denx == -1)] = (num22x / -denxmodificat)[np.where(deny - denx == -1)]

        denymodificat = np.where(deny - denx == 1, deny, 1)  # per evitar problemes interns dividint entre 0
        f21[np.where(deny - denx == 1)] = (num21y / denymodificat)[np.where(deny - denx == 1)]
        f11[np.where(deny - denx == 1)] = (num11y / -denymodificat)[np.where(deny - denx == 1)]

        B = (f11 * reference_image[A11[:, 0], A11[:, 1]] + f12 * reference_image[A12[:, 0], A12[:, 1]] + f21 *
             reference_image[
                 A21[:, 0], A21[:, 1]] + f22 * reference_image[A22[:, 0], A22[:, 1]])

        '''
        Això és el que vaig fer inicialment però no és realment interpolació bilineal 
         # SI TOT FOSSIN NO ENTERS FUNCIONA:
        f11 = (1 - (-A11[:, 0] + A[:, 0]) * (A11[:, 1] - A[:, 1])) / 3
        f12 = (1 - (A12[:, 0] - A[:, 0]) * (A12[:, 1] - A[:, 1])) / 3
        f21 = (1 - (-A21[:, 0] + A[:, 0]) * (-A21[:, 1] + A[:, 1])) / 3
        f22 = (1 - (A22[:, 0] - A[:, 0]) * (-A22[:, 1] + A[:, 1])) / 3

        #canvis pels enters:
        # canviam les fij de les posicions on les COLUMNES són enters:
        f11[np.where(A11[:, 1] == A[:, 1])] = (1 - (-A11[:, 0] + A[:, 0]))[np.where(A11[:, 1] == A[:, 1])]
        f12[np.where(A11[:, 1] == A[:, 1])] = (1 - (A12[:, 0] - A[:, 0]))[np.where(A11[:, 1] == A[:, 1])]
        f21[np.where(A11[:, 1] == A[:, 1])] = 0
        f22[np.where(A11[:, 1] == A[:, 1])] = 0

        # canviam les fij de les posicions on les FILES són enters:
        f11[np.where(A11[:, 0] == A[:, 0])] = (1 - (A11[:, 1] - A[:, 1]))[np.where(A11[:, 0] == A[:, 0])]
        f12[np.where(A11[:, 0] == A[:, 0])] = 0
        f21[np.where(A11[:, 0] == A[:, 0])] = (1 - (-A21[:, 1] + A[:, 1]))[np.where(A11[:, 0] == A[:, 0])]
        f22[np.where(A11[:, 0] == A[:, 0])] = 0

        # si tant files com columnes són enters:
        f11[np.where(np.sum(A11 == A, axis=1) == 2)] = 1
        f12[np.where(np.sum(A11 == A, axis=1) == 2)] = 0
        f21[np.where(np.sum(A11 == A, axis=1) == 2)] = 0
        f22[np.where(np.sum(A11 == A, axis=1) == 2)] = 0

        '''

        return B