import numpy as np
import cv2


class ObjectExtraction:
    """
        A class used to perform Object Extraction from Image

        Attributes
        ----------
        width : int
            width of the object to be extracted
        height : int
            height of the object to be extracted
        source_image : numpy.ndarray
            Source Image

        Methods
        -------
        extract(source_image) [public]
            extract object from the image provided.
        __preprocess() [private]
            Preprocess Source Image.
        __get_contour_of_largest_object(image)[private]
            Find the Contour of the largest object.
        __perspective_transform()[private]
            Transform the source image to the required output object size/shape
        __calculate_alpha_channel()[private]
            Find out the alpha channel based on the contour
    """

    def __init__(self, width: int, height: int, canny_lower=100, canny_upper=200):
        """
            Initialize the ObjectExtraction class instance

            Parameters
            ----------
            width : int
               The name of the animal
            height : int
              height The sound the animal makes
            canny_lower : int
              Lower threshold for Canny Edge detector
            canny_upper : int
              Upper threshold for Canny Edge detector

        """
        self.width = width
        self.height = height
        self.source_image = None
        self.canny_lower = canny_lower
        self.canny_upper = canny_upper

    def __preprocess(self):
        """
            Preprocess Source Image before finding contours

            Parameters
            ----------

            Returns
            -------
            edge:numpy.ndarray
                Processed image as numpy.ndarray
        """

        # Convert to Gray Image
        image_gray = cv2.cvtColor(self.source_image, cv2.COLOR_BGR2GRAY)

        # Use bilateralFilter() to clean the image. Bilateral Filter
        # is an edge preserving blur filter.
        image_gray = cv2.bilateralFilter(image_gray, 100, 20, 20)

        # Detect the edges using Canny Edge Detector
        edge = cv2.Canny(image_gray, self.canny_lower, self.canny_upper)

        # Alternatively cv2.threshold() can be used in place of Edge Detection
        # However performing edge detection before finding Contours works better than
        # image threshold.
        # ret, thresh = cv2.threshold(image_gray, 127, 255, 0)

        # Return the preprocessed image
        return edge

    def __get_contour_of_largest_object(self, image):
        """
            Find the Contour of the largest object.

            Parameters
            ----------
            image : numpy.ndarray
                Processed image

            Returns
            -------
            contour:numpy.ndarray
                The largest contour
        """

        # Find the Contours using findContours() function. Use RETR_EXTERNAL for getting only outer contours
        # and CHAIN_APPROX_SIMPLE for the end points of the contour.
        _, contours, _ = cv2.findContours(image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        # Sort the contours from larger to smaller by contour area and get the first contour (largest one).
        contour = sorted(contours, key=cv2.contourArea, reverse=True)[0]

        return contour

    def __perspective_transform(self, contour):
        """
            Transform the source image to the required output object size/shape.

            Parameters
            ----------
            image : numpy.ndarray
                Processed image

            Returns
            -------
            object:numpy.ndarray
                The extracted object
            transformation_matrix:numpy.ndarray
                The transformation matrix M is returned for more processing
        """

        # Get the 4 Points that can explain the contour of the object
        rect = cv2.minAreaRect(contour)

        # Convert the points to pixels
        box = cv2.boxPoints(rect)

        # Convert values from Float to Int
        box_int = np.int0(box)

        # Get the Contour Area
        areaCnt = cv2.contourArea(contour)

        # Get the Rectangle Area
        areaBox = cv2.contourArea(box_int)

        # Validate if the captured contour is a rectangle
        valid = areaCnt / areaBox > 0.95

        if valid:
            # extract rect object in tuples
            ((x, y), (w, h), angle) = rect

            # if width is greater than height
            # Need this step so that the card will always be in portrait mode.
            if w > h:
                # Define the transformed object
                transformed_object = np.array([[0, 0],
                                               [self.width, 0],
                                               [self.width, self.height],
                                               [0, self.height]], dtype=np.float32)
            else:
                transformed_object = np.array([[self.width, 0],
                                               [self.width, self.height],
                                               [0, self.height],
                                               [0, 0]], dtype=np.float32)

            # Calculate the transformation matrix
            transformation_matrix = cv2.getPerspectiveTransform(src=box, dst=transformed_object)

            # Extract the object from the source image
            object = cv2.warpPerspective(self.source_image, transformation_matrix, (self.width, self.height))

            return object, transformation_matrix
        else:
            return None, None

    def __calculate_alpha_channel(self, contour, transformation_matrix):
        """
            Calculate the alpha channel using the contour

            Parameters
            ----------
            contour : numpy.ndarray
                The largest contour
            transformation_matrix:numpy.ndarray
                The transformation matrix M

            Returns
            -------
            alpha_channel:numpy.ndarray
                The created alpha channel

        """

        # Reshape the contour from (n,1,2) - > (1,n,2)
        # Also convert it to float32
        contour = contour.reshape(1, -1, 2).astype(np.float32)

        # Apply the same perspective transformation
        # Convert the output to int
        contour_warp = cv2.perspectiveTransform(contour, transformation_matrix).astype(int)

        # Create an ndarray with all zeros and same width/height as the output object
        alpha_channel = np.zeros((self.height, self.width), dtype=np.uint8)

        # Fill the alpha channel with the contour
        cv2.drawContours(image=alpha_channel, contours=contour_warp, contourIdx=-1, color=(255, 255, 255), thickness=cv2.FILLED)

        return alpha_channel

    def extract(self, source_image):
        """
            Extract the largest object from the Source Image

            Parameters
            ----------
            source_image : numpy.ndarray
                Processed image

            Returns
            -------
            object:numpy.ndarray
                The extracted object

        """

        self.source_image = source_image

        # Preprocess the image before finding contours
        preprocessed_image = self.__preprocess()

        # Get the contour of the largest object
        contour = self.__get_contour_of_largest_object(preprocessed_image)

        # Extract the object and transformation matrix (M) from the source image
        # using perspective transformation
        object, transformation_matrix = self.__perspective_transform(contour)

        # Verify whether detection is successful
        if object is not None:

            # Calculate the alpha channel as per the contour
            alpha_channel = self.__calculate_alpha_channel(contour, transformation_matrix)

            # Add alpha channel to the object
            object = cv2.cvtColor(object, cv2.COLOR_BGR2BGRA)

            # Update the alpha channel with the derived alpha_channel
            object[:, :, 3] = alpha_channel

            return object
        else:
            return None
