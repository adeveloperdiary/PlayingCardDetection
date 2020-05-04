import numpy as np
import cv2
import os
import colorsys
import random
from tqdm import tqdm


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

    def __init__(self, yolo_config, confidence_factor=0.01, threshold=0.3, enable_gpu=False):
        """
            Initializer method for YOLODetection implementation class.

            :parameter
            yolo_config : dict
                - A dict object to pass all the yolo related parameters.
                    - weights : path of the weights of the network
                    - cfg     : path of the cfg file of the darknet model using in training
                    - labels  : path of the label names file used in training.
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

        # parse the file to create an array of class labels
        self.CLASS_LABELS = open(yolo_config['labels']).read().strip().split(os.linesep)

        self.confidence_factor = confidence_factor
        self.threshold = threshold
        self.enable_gpu = enable_gpu

        # assign random colors for each class labels
        self.COLORS = self._load_colors()

        # find input dimension of the network
        self.network_input_dim = self._find_network_input_dim()

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

    def _find_network_input_dim(self):
        """
            Read the input dimension from the cfg file

            :returns
            width : string
                width of the input image to the network
            height : string
                height of the input image to the network
        """
        width = None
        height = None

        # read the file and parse it as an array
        cfg = open(self.YOLO_CFG).read().strip().split(os.linesep)

        # Loop through each line
        for line in cfg:

            # If both with and height have been found then return them.
            if width is not None and height is not None:
                return width, height

            # Find the width and height if the line is not a comment
            if "width" in line and '#' not in line and '=' in line:
                width = int(line[line.index('=') + 1:])
            elif "height" in line and '#' not in line and '=' in line:
                height = int(line[line.index('=') + 1:])

        if width is None or height is None:
            raise Exception('[ERROR] Reading cfg file ... ')

    def _load_yolo_network(self):
        """
            Loads the YOLO Network using opencv dnn module.
        """

        # load the model
        self.network = cv2.dnn.readNetFromDarknet(yolo_config_path, yolo_weights_path)

        # Enable GPU if needed ( need to install opencv from source, also
        # CUDA supported GPU )
        if self.enable_gpu:
            self.network.setPreferableBackend(cv2.dnn.DNN_BACKEND_CUDA)
            self.network.setPreferableTarget(cv2.dnn.DNN_TARGET_CUDA)

        # Get the name of the output layers
        self.layers_name = self.network.getLayerNames()
        self.layers_name = [self.layers_name[i[0] - 1] for i in self.network.getUnconnectedOutLayers()]

    def _forward_pass(self, image):
        """
            Pass the image through the network once and capture the result

            :parameter
            image : ndarray
                - the image array
        """

        # read the image height and width
        self.H, self.W, _ = image.shape

        # load the network
        blob = cv2.dnn.blobFromImage(image, 1 / 255.0, self.network_input_dim, swapRB=True, crop=False)

        # set input image
        self.network.setInput(blob)

        # Run the forward pass
        self.layer_outputs = self.network.forward(self.layers_name)

    def _select_bounding_boxes(self):
        """
            Select the bounding boxes based on the confidence factor
            then run non max suppression.

            :parameter
            image : ndarray
                - the image array
        """
        self.bounding_boxes = []
        self.confidences = []
        self.class_ids = []

        # Loop through the output layers
        for output in self.layer_outputs:
            # Loop through each detection
            for detection in output:

                # save the score, class id and confidence factor
                # The first 4 values are center x, center y, width, height,
                # remaining values are the scores for each class labels.
                scores = detection[5:]

                # get the class with the height score
                class_id = np.argmax(scores)

                # Find the prediction confidence
                confidence = scores[class_id]

                # if the prediction confidence is more than confidence factor
                # add the details to the array
                if confidence > self.confidence_factor:
                    # calculate the center based on width of the image
                    center_x = int(detection[0] * self.W)

                    # calculate the center based on height of the image
                    center_y = int(detection[1] * self.H)

                    # calculate the width and height
                    width = int(detection[2] * self.W)
                    height = int(detection[3] * self.H)

                    # Find x,y using center, width and height
                    x = int(center_x - width / 2)
                    y = int(center_y - height / 2)

                    # saves the details in array
                    self.bounding_boxes.append([x, y, width, height])
                    self.confidences.append(float(confidence))
                    self.class_ids.append(class_id)

        # run non max suppression on the selected bounding boxes
        self.indexes = cv2.dnn.NMSBoxes(self.bounding_boxes, self.confidences, self.confidence_factor, self.threshold)

    def _draw_annotations(self, image):
        """
           Draw the object annotation using the bounding box and indexes
           returned by non max suppression

            :parameter
            image : ndarray
                - the image array

            :return
            annotated_image : ndarray
                - the annotated image output
        """
        font = cv2.FONT_HERSHEY_SIMPLEX

        # if there are identified objects
        if len(self.indexes) > 0:

            # Loop through all the indexes which need to be drawn
            for i in self.indexes.flatten():
                # Find the dimensions and color
                x, y, w, h = self.bounding_boxes[i]
                color = self.COLORS[self.class_ids[i]]

                # Draw the bounding box rectangle
                cv2.rectangle(image, (x, y), (x + w, y + h), color, 2)

                # Find the width/height of text annotation so thar we cn draw background of it.
                text = "{}: {}".format(self.CLASS_LABELS[self.class_ids[i]], str(int(self.confidences[i] * 100)) + "%")
                (label_width, label_height), _ = cv2.getTextSize(text, font, 0.5, 2)

                # Draw the text background and text
                cv2.rectangle(image, (x - 1, y - label_height - 10), (x + label_width, y), color, cv2.FILLED)
                cv2.putText(image, text, (x, y - 5), font,
                            0.5, [0, 0, 0], 2)
        return image

    def detect_objects_in_image(self, image_path, save_to=None, display=False):
        """
            Detects objects in images and annotates the image.

            :parameter
            image_path : string
                - Path of the image.
            save_to : string
                - Optional save location of the annotated image.
            display : string
                - Display the annotated image
        """

        # load the YOLO network
        self._load_yolo_network()

        # Read the input image
        image = cv2.imread(image_path)

        # Run a forward pass
        self._forward_pass(image)

        # Select the bounding boxes
        self._select_bounding_boxes()

        annotated_image = self._draw_annotations(image)

        if save_to:
            cv2.imwrite(save_to, annotated_image)

        if display:
            cv2.imshow("Annotated Image", annotated_image)
            cv2.waitKey(-1)
            cv2.destroyAllWindows()

    def detect_objects_in_videos(self, video_path, save_to=None, display=False, output_format="mp4v"):
        """
            Detects objects in videos and annotates each frame.

            :parameter
            video_path : string
                - Path of the video.
            save_to : string
                - Optional save location of the annotated video.
            display : string
                - Display the annotated frames
            output_format : string
                - Output video format
        """

        # video writer
        writer = None

        # load the YOLO network
        self._load_yolo_network()

        # Read the input video
        video_file = cv2.VideoCapture(video_path)

        # Find total number of frames
        try:
            total_frames = int(video_file.get(cv2.CAP_PROP_FRAME_COUNT))
        except:
            total_frames = -1

        # Create instance of progress bar
        pbar = tqdm(total=total_frames)

        # Loop until video is closed.
        while video_file.isOpened():
            # read each frame
            ret, frame = video_file.read()
            if ret:
                # Run a forward pass
                self._forward_pass(frame)

                # Select the bounding boxes
                self._select_bounding_boxes()

                # Draw the annotations
                annotated_frame = self._draw_annotations(frame)

                if save_to:
                    if writer is None:
                        # instantiate the video writer object
                        fourcc = cv2.VideoWriter_fourcc(*output_format)
                        writer = cv2.VideoWriter(save_to, fourcc, 30, (frame.shape[1], frame.shape[0]), True)

                    # write frame to file
                    writer.write(annotated_frame)

                if display:
                    # display each frame
                    cv2.imshow("video", annotated_frame)
                    cv2.waitKey(1)
                pbar.update(1)
            else:
                pbar.close()
                cv2.destroyAllWindows()
                video_file.release()
                break

    def detect_objects_in_cam(self, camera=0, save_to=None, output_format="mp4v"):
        """
            Detects objects in camera and annotates each frame.

            :parameter
            camera : int
                - Camera device number.
            save_to : string
                - Optional save location of the annotated video.
            output_format : string
                - Output video format
        """

        # video writer
        writer = None

        # load the YOLO network
        self._load_yolo_network()

        # Read the input video
        video_camera = cv2.VideoCapture(camera)

        # Loop until video is closed.
        while video_camera.isOpened():
            # read each frame
            ret, frame = video_camera.read()
            if ret:

                # Run a forward pass
                self._forward_pass(frame)

                # Select the bounding boxes
                self._select_bounding_boxes()

                # Draw the annotations
                annotated_frame = self._draw_annotations(frame)

                if save_to:
                    if writer is None:
                        # instantiate the video writer object
                        fourcc = cv2.VideoWriter_fourcc(*output_format)
                        writer = cv2.VideoWriter(save_to, fourcc, 30, (frame.shape[1], frame.shape[0]), True)

                    # write frame to file
                    writer.write(annotated_frame)

                    # display each frame

                cv2.imshow("video", annotated_frame)
                key = cv2.waitKey(1)
                if key == 27:
                    cv2.destroyAllWindows()
                    video_camera.release()
                    break
            else:
                cv2.destroyAllWindows()
                video_camera.release()
                break


if __name__ == '__main__':
    yolo_weights_path = "../yolo/yolov3-cards_11000.weights"
    yolo_config_path = "../yolo/yolov3-cards.cfg"
    class_labels = '../yolo/cards.names'

    detection = YOLODetection(
        yolo_config={'weights': yolo_weights_path,
                     'cfg': yolo_config_path,
                     'labels': class_labels})

    detection.detect_objects_in_image(image_path='../input/title.jpg', save_to="../output/out1.jpg", display=True)
    # detection.detect_objects_in_videos(video_path='../input/video.mov', save_to="../output/out.mov", display=False)
    # detection.detect_objects_in_cam(save_to='../output/cam.mov')
