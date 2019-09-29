
import numpy as np
from matplotlib import pyplot as plt

random_image =np.random.random([500,500])#matriu de 500x500 amb valors aleatoris


#representam imatges en blanc i negre
plt.imshow(random_image, cmap='gray')
#plt.show ()

from skimage import data

coins = data.coins()

print('Type:', type(coins))
print('dtype:',coins.dtype)
print('shape:',coins.shape)

plt.imshow(coins, cmap='gray')
#plt.show()això segons els videos no hauria de ser necessari pero nomes amb imshow no mo mostra

#imatges a color son arrays 3D on la darrera dimensio te tamany
# tres i representa els canals vermell,verd i blau

cat = data.chelsea()
print('Shape:',cat.shape)
print('Valors min/max', cat.min(),cat.max())

plt.imshow(cat)
#plt.show()per veure la imatge del moix


#canviam la imatge del moix
cat[10:110,10:110,:]=[255,0,0]#de la imatge cat de la fila i columnes 10 a 110 canviam el valor de tots els pixels per vermell
plt.imshow(cat)
#plt.show()per veure la imatge del moix amb el quadrat vermell


#displaying images using matplotlib
img0 = data.chelsea()
img1 = data.rocket()

f, (ax0, ax1) = plt.subplots(1, 2, figsize = (20,10))
#deim que els grafics les faci a una fila i 2 columnes

ax0.imshow(img0)
ax0.set_title('Cat', fontsize= 18)
ax0.axis('off')#sense eixos per la imatge del moix

ax1.imshow(img1)
ax1.set_title('Rocket', fontsize= 18)
ax1.set_xlabel(r'Launching position $\alpha =320$')

ax1.vlines([202,300],0,img1.shape[0], colors = 'magenta',
           linewidth=2, label ='Side tower position')
ax1.plot([168,190,200],[400,200,300], color='white',
         linestyle='--', label='Side angle')
#feim una linia discontinua que a nes 168 vagi a nes 400, es190 a nes 200 i es 200 a nes 300

ax1.legend();



#data types and image values

#feim dues imatges iguals d'escala de grisos pero una els colors van de 0 a 255 i l'altre de 0 a 1

linear0 = np.linspace(0,1,2500).reshape((50,50))
linear1 = np.linspace(0,255,2500).reshape((50,50)).astype(np.uint8)

print('Linear0',linear0.dtype,linear0.min(),linear0.max())
print('Linear1',linear1.dtype,linear1.min(),linear1.max())

fig, (ax0, ax1)= plt.subplots(1, 2, figsize = (15,15))
ax0.imshow(linear0, cmap='gray')
ax1.imshow(linear1, cmap='gray');

#plt.show()

#IMAGE I/O

from skimage.io import imread
from spline_registration.utils import get_databases_path

imatgefar=imread('/Users/mariamagdalenapolpujadas/Desktop/Desktop/universitat/tfg/GITHUB/tutorials/far.png')

#enumerate

animals = ['moix', 'ca', 'lleo']
for i, animals in enumerate(animals):
    print('l´ animal de la posicio {} és un {}'.format(i,animals))




