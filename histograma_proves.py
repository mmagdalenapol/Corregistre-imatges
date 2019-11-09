from skimage.io import imread
import numpy as np
from matplotlib import pyplot as plt
from skimage import data
cat = data.chelsea()
rangcolors=np.linspace(cat[:,:].min(),cat[:,:].max(),10)
print(rangcolors,cat.shape)
rangR= np.ones((cat.shape))
rangG= np.ones((cat.shape))
rangB= np.ones((cat.shape))
for fila in range (0,cat.shape[0]):
    for columna in range (0,cat.shape[1]):
       #per R
        a = rangcolors-cat[fila,columna][0]
        for i in range (0,len(a)):
            if a[i] >= 0 :
                if i==0:
                    rangR[fila,columna] = [rangcolors[i],0,0]
                    i=len(a)+1
                else:
                    if a[i-1]<0:
                        rangR[fila, columna] = [rangcolors[i-1], 0, 0]
                        i = len(a) + 1

         # per G
        a = rangcolors - cat[fila, columna][1]
        for i in range(0, len(a)):
            if a[i] >= 0:
                if i == 0:
                    rangG[fila, columna] = [0, rangcolors[i], 0]
                    i = len(a) + 1
                else:
                    if a[i - 1] < 0:
                        rangG[fila, columna] = [0, rangcolors[i-1 ], 0]
                        i = len(a) + 1
        # per B
        a = +rangcolors - cat[fila, columna][2]
        for i in range(0, len(a)):
            if a[i] >= 0:
                if i == 0:
                    rangB[fila, columna] = [0, 0, rangcolors[i]]
                    i = len(a) + 1
                else:
                    if a[i - 1] < 0:
                        rangB[fila, columna] = [0, 0, rangcolors[i-1]]
                        i = len(a) + 1