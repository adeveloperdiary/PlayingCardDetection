import pickle
import random
import cv2


class Objects:
    def get_random(self, name=None, display=False):
        pass

    def display(self, image, contours=[]):
        # If Alpha channel exists then apply the mask
        if image.shape[2] == 3:
            image = cv2.bitwise_or(image, image, mask=image[:, :, 3])
        for contour in contours:
            cv2.drawContours(image, [contour], 0, (0, 255, 0), 2)

        cv2.imshow("image", image)
        cv2.waitKey(0)


class Cards(Objects):
    def __init__(self, path):
        self._cards = pickle.load(open(path, 'rb'))
        self._num_cards_per_name = {k: len(self._cards[k]) for k in self._cards}
        print(self._num_cards_per_name)

    def get_random(self, name=None, display=False):
        if name is None:
            name = random.choice(list(self._cards.keys()))

        card, hull1, hull2 = self._cards[name][random.randint(0, self._num_cards_per_name[name] - 1)]
        card = cv2.imread(card, -1)
        card = cv2.cvtColor(card, cv2.COLOR_BGRA2RGBA)
        if display:
            self.display(card, [hull1, hull2])

        return card, name, [hull1, hull2]


# input_dir = "/Volumes/Samsung_T5/datasets/cards_ref.pck"

# cards = Cards(path=input_dir)
# cards.get_random(display=True)
from glob import glob


class Backgrounds(Objects):
    def __init__(self, path):
        self.dirs = glob(path)

    def get_random(self, name=None, display=False):
        file = self.dirs[random.randint(0, len(self.dirs) - 1)]
        image = cv2.imread(file)
        if display:
            self.display(image)
        return image

# bg = Backgrounds(path="/Volumes/Samsung_T5/datasets/dtd/images/*/*.jpg")
# bg.get_random(display=True)
