import pickle
import random
import cv2


class Objects:
    """
        An abstract class used to define the generation process of the objects.
            - The Scene class uses this to generate random objects and backgrounds.

        Attributes
        ----------
        path : string
            location of the image files

        Methods
        -------
        get_random(name,display) [public]
            - pick a random object from the path
        display(image,contours) [public]
            - display the image with the contours
    """

    def get_random(self, name=None, display=False):
        """
            abstract method for getting random object

            Parameters
            ----------
            name : string
               Get specific type of object (optional).
            display : bool
              display the selected object
        """
        pass

    def display(self, image, contours=[]):
        """
            display the image with the contours

            Parameters
            ----------
            image : array
               Image array to be displayed
            contours : list [optional]
              highlight the contours.
        """
        # If Alpha channel exists then apply the mask
        if image.shape[2] == 3:
            image = cv2.bitwise_or(image, image, mask=image[:, :, 3])
        # Loop through the contours and draw them accordingly
        for contour in contours:
            cv2.drawContours(image, [contour], 0, (0, 255, 0), 2)

        cv2.imshow("image", image)
        cv2.waitKey(0)


class Cards(Objects):
    """
        An implementation class for cards object

        Attributes
        ----------
        path : string
            location of the image files

        Methods
        -------
        get_random(name,display) [public]
            - pick a random card from the pickle file.
    """

    def __init__(self, path):
        """
            Constructor for the Cards class.
                - Loads the cards from the pickle object.
                - Find how many types of cards are available.
                - The pickle dump does not have the actual images.
                  rather just the path of the source image and
                  the convex hulls.
                - This way the program loads faster, takes less memory
                  and the size of the pickle file remains low.

            Parameters
            ----------
            path : string
               Path of the pickle object
        """
        self._cards = pickle.load(open(path, 'rb'))
        self._num_cards_per_name = {k: len(self._cards[k]) for k in self._cards}
        print(self._num_cards_per_name)

    def get_random(self, name=None, display=False):
        """
            - Picks a card randomly from the list of cards.
            - Reads the image from fs (retains the alpha channel).
            - Converts the image from BGRA (opencv) -> RGBA (standard)
            - If display is True then shows the image by calling display()
            - name can be used to pick specific objects.

            Parameters
            ----------
            name : string (optional)
               name of the card type

            Returns
            ----------
            card : ndarray
               The Card image
            name : string
                Class of the selected card
            hulls : list
                List of 2 convex hulls

        """
        if name is None:
            name = random.choice(list(self._cards.keys()))

        card, hull1, hull2 = self._cards[name][random.randint(0, self._num_cards_per_name[name] - 1)]

        # Preserve the alpha channel
        card = cv2.imread(card, -1)

        # Convert the color channels
        card = cv2.cvtColor(card, cv2.COLOR_BGRA2RGBA)
        if display:
            self.display(card, [hull1, hull2])

        return card, name, [hull1, hull2]


from glob import glob


class Backgrounds(Objects):
    """
        An implementation class for background object.

        Attributes
        ----------
        path : string
            directory of the image files

        Methods
        -------
        get_random(name,display) [public]
            - pick a random background from path.
    """

    def __init__(self, path):
        """
            Constructor for the Backgrounds class.
                - reads all the files in the path

            Parameters
            ----------
            path : string
               Path of the directly. Define the parent dir with
               file ext. ex - /datasets/dtd/images/*/*.jpg
        """
        self.dirs = glob(path)

    def get_random(self, name=None, display=False):
        """
            - Picks a background randomly from the path.
            - Reads the image from fs

            Parameters
            ----------
            None

            Returns
            ----------
            image : ndarray
               The Background image

        """
        file = self.dirs[random.randint(0, len(self.dirs) - 1)]
        image = cv2.imread(file)
        if display:
            self.display(image)
        return image
