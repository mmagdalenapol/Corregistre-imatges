from skimage.transform import resize


class BaseTransform:
    def find_best_transform(self, reference_image, input_image):
        raise NotImplementedError

    def apply_transform(self, input_image):
        raise NotImplementedError

    def visualize_transform(self):
        return None


class Rescala (BaseTransform):
    def __init__(self):
        self.dim_imatge = None

    def find_best_transform(self, reference_image, input_image):
        self.dim_imatge = reference_image.shape
        return resize(input_image,self.dim_imatge)

    def apply_transform(self, input_image):
        return input_image

