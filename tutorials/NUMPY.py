import numpy as np
import matplotlib.pyplot as plt
import time
import sys

a = np.array( [1,2,3] )
b = np.array( [(1,2,3),(4,5,6)])

print (a)
print (b)


#mostra de que numpy ocupa menys memoria que una llista
S=range(1000) #interval entre 0 i 1000 fet amb una llista

print (sys.getsizeof(5)*len(S))
#aixo em diu la memoria que ocupa la meva llista ja que
#sys.getsizeof(5) em dona la memoria que ocupa un element i
#com ho multiplicam per la longitut de la llista tenim la memoria total.

D = np.arange(1000)
# interval entre 0 i 1000 fet amb numpy
print(D.size* D.itemsize)
#aixo es la memoria ocupada per l'array numpy que veim que
#es mes petita que la ocupada per una llista normal


#mostra de que numpy es mes rapid que una llista

SIZE = 1000000

L1 = range(SIZE)
L2 = range(SIZE)

A1 = np.arange(SIZE)
A2 = np.arange(SIZE)

start= time.time()

result = [(x,y) for x,y in zip(L1,L2)]

print((time.time()-start)*1000000)
#em dona el temps que necessita la llista per fer la suma

start = time.time()

result = A1 + A2

print((time.time()-start)*1000000)
#em dona el temps que necessita l'array numpy per fer la suma
#veim que efectivament numpy tarda menys en fer el mateix


#OPERACIONS

a = np.array([(1,2,3),(2,3,4)]) #2 files 3 columnes

print(a.ndim) #em dona la dimensio que efectivament es 2
print(a.itemsize) #em diu el que ocupa cada element de a en bytes
print(a.dtype) #tipo de data
print(a.size)#nombre d'elements del meu array.
print(a.shape)#hem diu les files,columnes del nostre array

a = a.reshape(3,2)
#reorganitza l'array a de manera que tengui 3 files i 2 columnes
print(a)

a = np.array([(1,2,3),(2,3,4),(5,6,7)])
print(a[0,2], a[0:,2],a[0:2,0])
#treim de la fila 0 el tercer element i de totes les files
# des de la 0 el tercer elelment
# des de la fila 0 a la tercera fora incloure el primer element

a = np.linspace (1,3,5)
#5 valors que estan equiespaiats entre 1 i 3
print(a)

a = np.array([1,2,3])
print(a.min(),a.max(),a.sum())


a=np.array([(1,2,3),(3,4,5)])
#sumar els elements d'una mateixa columna:
print(a.sum(axis=0))

#sumar els elements d'una mateixa fila:
print(a.sum(axis=1))


#trobar arrels quadrades i desviacions típiques de matrius
print(np.sqrt(a)) #arrel de cada element
print(np.std(a)) #com es separa cada element de la mitjana

#suma,resta,multiplicacio i divisio de matrius element a element

a=np.array([(1,2,3),(3,4,5)])
b=np.array([(1,2,3),(3,4,5)])

print(a+b)
print(a-b)
print(a*b)
print(a/b)


#posar una matriu davall l'altre
print(np.vstack((a,b)))
#si volem posar una matriu devora l'altre feim:
print(np.hstack((a,b)))

#convertir la matriu en un vector
print(a.ravel())

#aixo utilitza matplotlib
#funcions elementals

x = np.arange(0,3*np.pi,0.1)
y = np.sin(x)
plt.plot(x,y)#fa el grafic del sinus
plt.show()#mostra el grafic del sinus

y = np.cos(x)
plt.plot(x,y)
plt.show()

y = np.tan(x)
plt.plot(x,y)
plt.show()


ar = np.array ([1,2,3])
print(np.exp(ar))#exponencial
print(np.log(ar))#ln
print(np.log10(ar))

