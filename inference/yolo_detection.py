import numpy as np
import cv2
import os
import colorsys
import random


class YOLODetection:
    """
        An implementation class for loading YOLO models using
        opencv dnn module and use it for inference.

        Methods
        -------
        __init__(yolo_config, media_path, media_type,
                confidence_factor,threshold,enable_gpu) [constructor]:
            - initializes the class and required parameters
        _load_colors:
            - loads dynamic colors for each class types.

    """

    def _load_colors(self):
        """
            Randomly selects colors for each class types. Use the HSV color space
            to select specific color variations.

            :return
            colors: list
                - List of different colors ( one for each class types )

        """
        colors = []

        # Loop through all the class labels.
        for i in range(len(self.CLASS_LABELS)):
            # First select a color in hsv color space.
            #  The text color will be black and the blue color is
            #  very dark to have black text color on top of it.
            #  Hence in order to not choose any absolute blue color,
            #  choose the hue from 0.05 to 0.95 randomly.
            #  We want bright vibrant color, hence choose the saturation
            #  from 0.5 to 1.0 and value to 255. We dont want to maximize the
            #  value parameter, otherwise everything will be white.
            # Then convert the hsv color space to rbg color space
            # and store in an array. This way we have one unique color for each class labels
            colors.append(colorsys.hsv_to_rgb(random.uniform(.05, .95), random.uniform(.5, 1), 255))
        return colors

    def __init__(self, yolo_config, media=None, media_type='image',
                 confidence_factor=0.01, threshold=0.3, enable_gpu=False):
        """
            Initializer method for YOLODetection implementation class.

            :parameter
            yolo_config : dict
                - A dict object to pass all the yolo related parameters.
                    - weights : path of the weights of the network
                    - cfg     : path of the cfg file of the darknet model using in training
                    - labels  : path of the label names file used in training.
            media : string
                - Media file. Path in case video or image.
            media_type : string
                - Type of media. Supported values are image/video/camera
            confidence_factor : int
                - Min confidence factor for each prediction needed to be annotated
            threshold : int
                - Threshold value for Non-Maximum Suppression. This is needed as YOLO
                  does not perform non-max suppression.
            enable_gpu : bool
                - Run inference in GPU is enabled ( need to build opencv from source )
        """
        self.YOLO_WEIGHTS = yolo_config['weights']
        self.YOLO_CFG = yolo_config['cfg']
        self.CLASS_LABELS = open(yolo_config['labels']).read().strip().split(os.linesep)
        self.COLORS = self._load_colors()
