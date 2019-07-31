

class MyRegistrationModel:

    def register(self, reference_image, input_image):
        raise NotImplementedError


class Transformada10x10:

    def __init__(self):
        self.dictionary_points = {}

    def register(self, reference_image, input_image):
        self.dictionary_points = {
            (0, 0): (x1, y1),
            (0, 1): (y2, x2),
            # ...
        }


