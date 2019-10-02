import numpy as np

def always_return_zero(reference_image, transformed_image):

    return 0

def SSD(reference_image, transformed_image):

    N = reference_image.shape[0]
    a = reference_image-transformed_image

    SSD = np.sum(a * a) / N

    return SSD

