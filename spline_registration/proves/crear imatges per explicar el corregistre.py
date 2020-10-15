from spline_registration.utils import visualitza_malla
from skimage.io import imsave, imread
import numpy as np
from spline_registration.transform_models import ElasticTransform_SSD
import random
from spline_registration.utils import create_results_dir
from spline_registration.losses import RMSE, info_mutua
import matplotlib.pyplot as plt

random.seed(1)
results_dir = create_results_dir(f'malla_inicial')
inp_img = imread('/Users/mariamagdalenapolpujadas/Desktop/universitat/tfg/GITHUB/spline_registration/proves/dog_input.jpg')
ref_img = imread('/Users/mariamagdalenapolpujadas/Desktop/universitat/tfg/GITHUB/spline_registration/proves/dog_reference.jpg')

mida_malla = np.asarray([[2, 2], [4, 4], [8, 8]])

dim_original = inp_img.shape
transform1 = ElasticTransform_SSD(mida_malla, dim_original)
iteracions = [2, 2, 2]

# com inicialitzar la malla i crear una imatge transformada a partir d'aquests valors
malla_vector_inicial = transform1.malla_inicial(0)
malla_inicial = transform1.parametres_a_malla(malla_vector_inicial,0)

malla_inicial_perturbada_x, malla_inicial_perturbada_y, coord_originals_x, coord_originals_y\
    = transform1.perturbar_malla_aleatoriament(malla_vector_inicial,inp_img,1,0)

img_tfd = transform1.transformar(inp_img,np.concatenate([malla_inicial_perturbada_x.ravel(),malla_inicial_perturbada_y.ravel()]),0)

visualitza_malla(img_tfd, malla_inicial[0], malla_inicial[1], 'Malla inicial sobre la imatge transformada', f'{results_dir}/malla_inicial')
visualitza_malla(inp_img, malla_inicial_perturbada_x, malla_inicial_perturbada_y, 'Malla destí sobre la imatge d´entrada', f'{results_dir}/malla_destí')

#Els millors paràmetres amb una malla 3 per 3 són:
malla3_vector = np.asarray([0.3797832, - 0.49126724, 1.98586629,21.51479479, 20.6428881,  23.11833813,
 42.3747391,  41.48904983, 43.97287403,  2.83224392, 23.39639463, 43.84613798,
 2.51048668, 23.08648065, 43.47963192, - 0.67858283, 19.83565137, 40.16052955])
malla3 = transform1.parametres_a_malla(malla3_vector,0)
malla3_vector = np.concatenate([dim_original[0] / 40 * malla3[0],dim_original[1] / 40 * malla3[1]]).ravel()
malla3 = transform1.parametres_a_malla(malla3_vector,0)

malla_original = transform1.parametres_a_malla(transform1.malla_inicial(0),0)
registered_image = transform1.transformar(inp_img, malla3_vector, 0)

RMSE3 = RMSE(ref_img,registered_image)
imsave(f'{results_dir}/imatge_registrada_{3, 3}_{RMSE3}.png',
               registered_image)

visualitza_malla(registered_image, malla_original[0], malla_original[1],
                 f'Malla inicial sobre la imatge transformada {3,3}',
                 f'{results_dir}/malla {3,3} sobre imatge registrada.png')

visualitza_malla(inp_img, malla3[0], malla3[1],
                 f'Malla destí sobre la imatge d´entrada {3,3}',
                 f'{results_dir }/malla {3,3} sobre la imatge d´entrada .png')



# Els millors paràmetres amb una malla 5 per 5 són:
malla5_vector = np.asarray([8.41679514e-01, - 2.21620225e-02,  2.36121290e-01,  3.47879119e-01,
 2.00627418e+00,  1.11924815e+01,  1.03503199e+01,  1.06552441e+01,
 1.08297643e+01,  1.24293263e+01,  2.14938474e+01, 2.07088630e+01,
 2.10030788e+01,  2.13002692e+01,  2.27239847e+01,  3.16301291e+01,
 3.09683024e+01,  3.13289584e+01,  3.18394498e+01,  3.30577326e+01,
 4.18645442e+01,  4.12451749e+01,  4.15897793e+01,  4.21966863e+01,
 4.32513963e+01,  1.69427351e+00,  1.22018192e+01,  2.28255736e+01,
 3.34540357e+01,  4.38594595e+01,  2.58972746e+00,  1.29011857e+01,
 2.32100862e+01,  3.35122735e+01,  4.37368073e+01,  1.59606218e+00,
 1.18127945e+01,  2.20003374e+01,  3.21624925e+01,  4.23315904e+01,
 - 4.02808949e-01,  1.00485792e+01,  2.05397044e+01,  3.10233783e+01,
 4.14229884e+01, - 2.10456379e+00,  8.28407811e+00,  1.86851945e+01,
 2.90734098e+01,  3.94146581e+01])
malla5 = transform1.parametres_a_malla(malla5_vector,1)
malla5_vector = np.concatenate([dim_original[0] / 40 * malla5[0], dim_original[1] / 40 * malla5[1]]).ravel()
malla5 = transform1.parametres_a_malla(malla5_vector,1)

malla_original = transform1.parametres_a_malla(transform1.malla_inicial(1),1)
registered_image = transform1.transformar(inp_img, malla5_vector, 1)

RMSE5 = RMSE(ref_img,registered_image)
imsave(f'{results_dir}/imatge_registrada_{5,5}_{RMSE5}.png',
               registered_image)
visualitza_malla(registered_image, malla_original[0], malla_original[1],
                 f'Malla inicial sobre la imatge transformada {5,5}',
                 f'{results_dir }/malla {5,5} sobre imatge registrada.png')

visualitza_malla(inp_img, malla5[0], malla5[1],                                   
                 f'Malla destí sobre la imatge d´entrada {5,5}',
                 f'{results_dir }/malla {5,5} sobre la imatge d´entrada .png')




# Els millora paràmetres amb una malla 9 per 9 són:

malla9_vector = np.asarray( [ 3.68029058e-01, -1.18954168e-01, -3.08606075e-01, -5.04242108e-02,
  1.24576880e-01, -1.81034468e-01, -8.21730035e-02,  4.88556776e-01,
  1.35284152e+00,  5.52390779e+00,  5.06440650e+00,  4.87310397e+00,
  5.10470451e+00,  5.29728223e+00,  5.05347418e+00,  5.12544720e+00,
  5.73120705e+00,  6.56600075e+00,  1.07147351e+01,  1.02985448e+01,
  1.00987535e+01,  1.02758170e+01,  1.04504702e+01,  1.03293799e+01,
  1.03642684e+01,  1.09795045e+01,  1.17748878e+01,  1.58258456e+01,
  1.54933542e+01,  1.52850363e+01,  1.53675693e+01,  1.55813405e+01,
  1.55482811e+01,  1.55929465e+01,  1.62206540e+01,  1.69430756e+01,
  2.09859596e+01,  2.06989225e+01,  2.04947651e+01,  2.04889439e+01,
  2.07241964e+01,  2.07240837e+01,  2.08427236e+01,  2.14639356e+01,
  2.20926471e+01,  2.60226922e+01,  2.58199574e+01,  2.56440766e+01,
  2.55730097e+01,  2.58667852e+01,  2.58164762e+01,  2.61228504e+01,
  2.66546623e+01,  2.71883396e+01,  3.10747596e+01,  3.09120193e+01,
  3.07952836e+01,  3.06889537e+01,  3.10662305e+01,  3.09321908e+01,
  3.14124504e+01,  3.19202769e+01,  3.23627785e+01,  3.61594938e+01,
  3.60427414e+01,  3.59524652e+01,  3.58506766e+01,  3.62590700e+01,
  3.60136669e+01,  3.67140483e+01,  3.71078317e+01,  3.74911761e+01,
  4.11215556e+01,  4.10392797e+01,  4.10044029e+01,  4.09031799e+01,
  4.13688166e+01,  4.10638219e+01,  4.18504898e+01,  4.22364223e+01,
  4.25727148e+01,  1.19019670e+00,  6.41048245e+00,  1.16801077e+01,
  1.69290652e+01,  2.21896312e+01,  2.74322261e+01,  3.26988962e+01,
  3.79139796e+01,  4.30840515e+01,  1.85365707e+00,  7.02991035e+00,
  1.22139088e+01,  1.73841827e+01,  2.25390104e+01,  2.77232900e+01,
  3.28864927e+01,  3.80711089e+01,  4.32039633e+01,  1.89767365e+00,
  7.11415923e+00,  1.23549072e+01,  1.75881264e+01,  2.28187095e+01,
  2.79881369e+01,  3.31846003e+01,  3.83125706e+01,  4.34438861e+01,
  1.18029799e+00,  6.44000022e+00,  1.17072802e+01,  1.69927879e+01,
  2.22244468e+01,  2.74348407e+01,  3.26130735e+01,  3.77326713e+01,
  4.28685866e+01,  9.98402883e-01,  6.14254370e+00,  1.12673940e+01,
  1.63870148e+01,  2.15268117e+01,  2.66971810e+01,  3.18613795e+01,
  3.70303062e+01,  4.21985043e+01,  3.67981724e-02,  5.21001903e+00,
  1.03732925e+01,  1.55044366e+01,  2.06586325e+01,  2.58250975e+01,
  3.10021883e+01,  3.61591717e+01,  4.13231846e+01, -9.67478037e-01,
  4.26842725e+00,  9.53785335e+00,  1.47890754e+01,  2.00233255e+01,
  2.52559239e+01,  3.04277570e+01,  3.56080664e+01,  4.07778486e+01,
 -1.61258107e+00,  3.59191810e+00,  8.82023158e+00,  1.40185586e+01,
  1.92565242e+01,  2.44840214e+01,  2.96669350e+01,  3.49072464e+01,
  4.01127972e+01, -3.09121021e+00,  2.19147996e+00,  7.50166478e+00,
  1.28350071e+01,  1.81046074e+01,  2.33671379e+01,  2.85526585e+01,
  3.37578333e+01,  3.89634014e+01])

malla9 = transform1.parametres_a_malla(malla9_vector,2)
malla9_vector = np.concatenate([dim_original[0] / 40 * malla9[0], dim_original[1] / 40 * malla9[1]]).ravel()
malla9 = transform1.parametres_a_malla(malla9_vector,2)

malla_original = transform1.parametres_a_malla(transform1.malla_inicial(2),2)
registered_image = transform1.transformar(inp_img, malla9_vector, 2)

RMSE9 = RMSE(ref_img,registered_image)
imsave(f'{results_dir}/imatge_registrada_{9,9}_{RMSE9}.png',
               registered_image)

visualitza_malla(registered_image, malla_original[0], malla_original[1],
                 f'Malla inicial sobre la imatge transformada  {9, 9}',
                 f'{results_dir }/malla {9, 9} sobre imatge registrada.png')

visualitza_malla(inp_img, malla9[0], malla9[1],
                 f'Malla destí sobre la imatge d´entrada {9, 9}',
                 f'{results_dir }/malla {9, 9} sobre la imatge d´entrada .png')









inp_img = imread('/Users/mariamagdalenapolpujadas/Desktop/universitat/tfg/GITHUB/spline_registration/proves/seagull_input.jpg')
ref_img = imread('/Users/mariamagdalenapolpujadas/Desktop/universitat/tfg/GITHUB/spline_registration/proves/seagull_reference.jpg')
img_tfd = transform1.transformar(inp_img,np.concatenate([malla_inicial_perturbada_x.ravel(),malla_inicial_perturbada_y.ravel()]),0)

visualitza_malla(img_tfd, malla_inicial[0], malla_inicial[1], 'malla inicial sobre la imatge transformada', f'{results_dir}/malla_inicial GAVINA')
visualitza_malla(inp_img, malla_inicial_perturbada_x, malla_inicial_perturbada_y, 'malla destí sobre la imatge d´entrada', f'{results_dir}/malla_destí GAVINA')

#Els millors paràmetres amb una malla 3 per 3 són:
malla3_vector = np.asarray( [ 1.25744029, -1.01717543, -0.04942233, 22.6796821,  20.46056566, 21.44160028,
 44.18960386, 42.03085542, 43.00684549, -0.32835092, 21.44832406, 43.32739782,
 -2.29628499, 19.47991129, 41.35106272, -1.81512232, 19.97386212,41.8591936 ])
malla3 = transform1.parametres_a_malla(malla3_vector,0)
malla3_vector = np.concatenate([dim_original[0] / 40 * malla3[0],dim_original[1] / 40 * malla3[1]]).ravel()
malla3 = transform1.parametres_a_malla(malla3_vector,0)

malla_original = transform1.parametres_a_malla(transform1.malla_inicial(0),0)
registered_image = transform1.transformar(inp_img, malla3_vector, 0)
IM3 = np.sum(info_mutua(ref_img, registered_image, 5))
imsave(f'{results_dir}/imatge_registrada_{3,3}_{IM3}.png',
               registered_image)

visualitza_malla(registered_image, malla_original[0], malla_original[1],
                 f'Malla inicial sobre la imatge transformada {3,3}',
                 f'{results_dir}/malla {3,3} sobre imatge registrada GAVINA.png')

visualitza_malla(inp_img, malla3[0], malla3[1],
                 f'Malla destí sobre la imatge d´entrada {3,3}',
                 f'{results_dir }/malla {3,3} sobre la imatge d´entrada GAVINA.png')

# Els millors paràmetres amb una malla 5 per 5 són:
malla5_vector = np.asarray([ 0.84886954,  0.53219011,-0.86821163, 0.138022  ,-0.17684077, 11.44422496,
 11.11922902,  9.73565297, 10.72993724, 10.44314623,21.89296535,21.55606505,
 20.21031012, 21.18577433, 20.95165676, 32.47728922,32.1270162 ,30.80768241,
 31.76156763, 31.57239542, 43.05531763, 42.69923339,41.39289406,42.34152859,
 42.17923561, -1.37166063,  9.80736302, 20.8961076 ,32.02788555,43.20587401,
 -1.639227  ,  9.53768004, 20.62271792, 31.75070868,42.9237038 ,-3.39290865,
  7.79725664, 18.91723416, 30.07447916, 41.27415826,-2.89668075, 8.30618961,
 19.4468132 , 30.62477017, 41.83101422, -2.23136128, 8.95706431,20.07572773,
 31.23024833, 42.42234994] )
malla5 = transform1.parametres_a_malla(malla5_vector,1)
malla5_vector = np.concatenate([dim_original[0] / 40 * malla5[0], dim_original[1] / 40 * malla5[1]]).ravel()
malla5 = transform1.parametres_a_malla(malla5_vector,1)

malla_original = transform1.parametres_a_malla(transform1.malla_inicial(1),1)
registered_image = transform1.transformar(inp_img, malla5_vector, 1)
IM5 = np.sum(info_mutua(ref_img, registered_image, 5))
imsave(f'{results_dir}/imatge_registrada_{5,5}_{IM5}.png',
               registered_image)
visualitza_malla(registered_image, malla_original[0], malla_original[1],
                 f'Malla inicial sobre la imatge transformada {5,5}',
                 f'{results_dir }/malla {5,5} sobre imatge registrada GAVINA.png')

visualitza_malla(inp_img, malla5[0], malla5[1],
                 f'Malla destí sobre la imatge d´entrada {5,5}',
                 f'{results_dir }/malla {5,5} sobre la imatge d´entrada GAVINA.png')

# Els millora paràmetres amb una malla 9 per 9 són:

malla9_vector = np.asarray(   [ 0.19671067,  0.10229209, -0.46239459, -1.02179205, -1.52292024, -0.97009256,
                              -0.610137   ,-0.45137559 ,-1.16699567 , 5.51029421 , 5.41527798 , 4.85980081,
                               4.31914349 , 3.80308594 , 4.35889821 , 4.70299953 , 4.84863278 , 4.15834,
                              10.83987754 ,10.74411946 ,10.19361095 , 9.67530511 , 9.09942579 , 9.6671946,
                               9.97332122 ,10.09251153 , 9.43852458 ,16.16229729 ,16.06600721 ,15.52841546,
                              15.04120902 ,14.39175803 ,14.97370166 ,15.23182377 ,15.3203084  ,14.72229804,
                              21.38125317 ,21.29461554 ,20.80587797 ,20.37777989 ,19.68040928 ,20.26985382,
                              20.48280344 ,20.53868637 ,20.01042692 ,26.7031465  ,26.62060891 ,26.15938744,
                              25.77026239 ,25.01162593 ,25.60304611 ,25.76153223 ,25.78248906 ,25.32701244,
                              31.95962698 ,31.88633864 ,31.45122019 ,31.09548877 ,30.28348049 ,30.86854368,
                              30.96941032 ,30.95221293 ,30.56534403 ,37.285023   ,37.21757913 ,36.79918202,
                              36.46543837 ,35.61072241 ,36.20054159 ,36.2621498  ,36.22136581 ,35.88773261,
                              42.57840506 ,42.51219862 ,42.0997889  ,41.77217256 ,40.89492799 ,41.47863304,
                              41.5185524  ,41.46498047 ,41.16128209 ,-1.99955341 , 3.64955    , 9.26893868,
                              14.88309516 ,20.49636381 ,26.09495473 ,31.70335829 ,37.32321772 ,42.90904366,
                              -2.41300589 , 3.26042469 , 8.92964459 ,14.61607216 ,20.31401643 ,26.00394779,
                              31.70037319 ,37.38750347 ,43.01077982 ,-2.62261812 , 3.01830904 , 8.62818633,
                              14.23991681 ,19.85032278 ,25.46304069 ,31.09863427 ,36.74237217 ,42.34921149,
                              -2.92864049 , 2.67857529 , 8.22930753 ,13.77529966 ,19.31364641 ,24.87692069,
                              30.48335777 ,36.11125879 ,41.71878507 ,-3.97843229 , 1.68057838 , 7.32875257,
                              12.98088705 ,18.59562421 ,24.21699957 ,29.85803016 ,35.5015927  ,41.12423782,
                              -4.14563921 , 1.51291837 , 7.15922942 ,12.80719618 ,18.4025543  ,24.01070868,
                              29.64065976 ,35.27755512 ,40.90999782 ,-3.29125438 , 2.35246561 , 7.97935373,
                              13.61766126 ,19.22126838 ,24.86074987 ,30.53601746 ,36.20562407 ,41.84587585,
                              -3.17067092 , 2.47237169 , 8.09526901 ,13.72065751 ,19.28744782 ,24.89205138,
                              30.53071082 ,36.17559282 ,41.81629376 ,-2.56973659 , 3.08636349 , 8.73379307,
                              14.38811539 ,19.98163324 ,25.60932743 ,31.26333168 ,36.91239898 ,42.55058647])

malla9 = transform1.parametres_a_malla(malla9_vector,2)
malla9_vector = np.concatenate([dim_original[0] / 40 * malla9[0], dim_original[1] / 40 * malla9[1]]).ravel()
malla9 = transform1.parametres_a_malla(malla9_vector,2)

malla_original = transform1.parametres_a_malla(transform1.malla_inicial(2),2)
registered_image = transform1.transformar(inp_img, malla9_vector, 2)
IM9 = np.sum(info_mutua(ref_img, registered_image, 5))
imsave(f'{results_dir}/imatge_registrada_{9,9}_{IM9}.png',
               registered_image)                               


visualitza_malla(registered_image, malla_original[0], malla_original[1],
                 f'Malla inicial sobre la imatge transformada  {9, 9}',
                 f'{results_dir }/malla {9, 9} sobre imatge registrada GAVINA.png')

visualitza_malla(inp_img, malla9[0], malla9[1],
                 f'Malla destí sobre la imatge d´entrada {9, 9}',
                 f'{results_dir }/malla {9, 9} sobre la imatge d´entrada GAVINA.png')




## imatges de la gavina en els 125 colors que la formen quan feim la informació mútua
inp_img = (inp_img - inp_img.min()) / (inp_img.max() - inp_img.min())
ref_img = (ref_img - ref_img.min()) / (ref_img.max() - ref_img.min())

inp_img_125 = np.floor_divide(inp_img, 1 / 5)
ref_img_125 = np.floor_divide(ref_img, 1 / 5)

imsave(f'{results_dir}/gavina input 125 colors.png', inp_img_125)
imsave(f'{results_dir}/gavina referencia 125 colors.png', ref_img_125)