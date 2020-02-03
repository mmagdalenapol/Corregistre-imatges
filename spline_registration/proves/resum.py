#3 febrer 2020
'''
El nostre objectiu és fer un corregistre elàstic entre dues imatges.
Per poder fer-ho el primer que feim és carregar les nostres imatges.
Així el primer que vaig fer va ser descobrir com donada una carpeta amb subcarpetes anar guardant les direccions de totes les imatges.
I així poder accedir-hi ràpid quan volguem fer feina amb una determinada imatge.
Això ho vaig fer al fitxer databases.

Si volem fer feina amb unes dades diferents a anhir el que hem de fer és
guardar la carpeta com una subcarpeta de databases. Ara tenim la funció anhir()
del fitxer databases que bàsicament ens retorna un diccionari amb amb keys les carpetes i valors els paths de les respectives imatges.

Amb una altre conjunt d'imatges podriem crear fàcilment una funció semblant que faci el mateix.
'''

from spline_registration.databases import anhir
data_anhir = anhir()
print(data_anhir.keys())

'''
Ara ja hem vist que tenim un diccionari amb keys els nom de les subcarpetes i
els valors els paths de cada una de les imatges que hi ha dins cada subcarpeta.
'''

'''
Un pic tenim les direccions el següent seria 
'''