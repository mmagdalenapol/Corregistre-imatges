
class BaseTransform:
    def __init__(self, loss_function):
        self.loss = loss_function

    def find_best_transform(self, reference_image, input_image):
        raise NotImplementedError

    def apply_transform(self, input_image):
        raise NotImplementedError

    def visualize_transform(self):
        return None


class DoNothingTransform(BaseTransform):
    def find_best_transform(self, reference_image, input_image):
        pass

    def apply_transform(self, input_image):
        return input_image


