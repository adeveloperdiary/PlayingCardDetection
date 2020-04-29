import numpy as np
import cv2
from dataset_creation.extract_object import ObjectExtraction
import glob
import os
from tqdm import tqdm
import random
import shutil


def get_alpha_mask(height, width):
    """
        Create Card Sized Alpha Mask

        Parameters
        ----------
        width : int
            width of the object to be extracted
        height : int
            height of the object to be extracted

        Returns
        -------
        alpha_mask:numpy.ndarray
            The Alpha Mask for the card
    """

    # Create an array as the height and width of the output card
    alpha_mask = np.ones((height, width), dtype=np.uint8) * 255

    # Draw a rectangle
    cv2.rectangle(alpha_mask, (0, 0), (width - 1, height - 1), 0, 2)

    # Draw a line at the 4 corners to make the rounded corner effect
    cv2.line(alpha_mask, (6, 0), (0, 6), 0, 2)
    cv2.line(alpha_mask, (width - 6, 0), (width, 6), 0, 2)
    cv2.line(alpha_mask, (0, height - 6), (6, height), 0, 2)
    cv2.line(alpha_mask, (width - 6, height), (width, height - 6), 0, 2)

    return alpha_mask


def capture_images_from_video(video_dir, output_dir, processed_dir, capture_in_every=1):
    """
        Captures card images from the video

        Parameters
        ----------
        video_dir : string
            Dir of the video files
        output_dir : string
            Output dir of the cards
        processed_dir : string
            Backup dir
        capture_in_every : int
            Which frame to read

        Returns
        -------
    """

    # Define the required Card width and Height
    card_width = 280
    card_height = 400

    # Create an instance of the ObjectExtraction() class
    image_extraction = ObjectExtraction(card_width, card_height, 10, 100)

    # Get the list of video files
    video_files = glob.glob(video_dir)

    # Loop through all the video files
    for i in range(len(video_files)):

        # Find the target class
        target_class = video_files[i].split("/")[-1].split(".")[0]

        # Create the target directory if does not exists
        if not os.path.exists(output_dir + "/" + target_class):
            os.mkdir(output_dir + "/" + target_class)

        # Create a pointer to the video file
        capture = cv2.VideoCapture(video_files[i])

        # Find total number of frames
        total_number_of_frames = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))

        # Find the frame width
        frame_width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))

        # Initialize total number of card captured
        total_card_captured = 0

        # Loop through total number of frames in the video
        # use tqdm for the progress bar
        for count in tqdm(range(total_number_of_frames), desc=target_class):

            # read each frame from the video
            ret, frame = capture.read()

            # Make sure the frame is available
            if ret:

                # find out whether to process the frame
                if count % capture_in_every == 0:

                    # Use Laplacian to determine the blurriness
                    # Not used here since there wasn't any blurred frame
                    # focus = cv2.Laplacian(frame, cv2.CV_64F).var()
                    # if focus < 100:
                    #    print("Focus too low", focus)

                    # Resize the frame if the size is more than 720
                    if frame_width > 720:
                        frame = cv2.resize(frame, (720, 405))

                    # Extract the card from the frame
                    card = image_extraction.extract(source_image=frame)

                    # If Card is detected then
                    if card is not None:

                        # Clean the mask using the rounder corner alpha mask
                        card[:, :, 3] = cv2.bitwise_and(card[:, :, 3], get_alpha_mask(card_height, card_width))

                        # Save the extracted card image to the disk
                        cv2.imwrite(
                            output_dir +
                            "/" + target_class + "/" + target_class + "_" + str(count) + "_" + str(random.randint(0, 100000)) + ".png",
                            card)

                        # Increment total card captured
                        total_card_captured += 1
            else:
                break

        # Release the video
        capture.release()

        print("Total # of card extracted :", total_card_captured)

        # Move the video file to backup folder
        if processed_dir is not None:
            shutil.move(video_files[i], processed_dir)


def display_image(output_dir, rect):
    """
        Display random extracted card images with bounding box

        Parameters
        ----------
        output_dir : string
            Output dir of the cards
        rect : tuple
            Bounding box

        Returns
        -------
    """

    # Get the list of image files
    files = glob.glob(output_dir + "/*/*.png")

    # Infinite Loop
    while True:
        # Choose a random image
        file = random.choice(files)

        # Read using opencv
        image = cv2.imread(file, cv2.IMREAD_UNCHANGED)

        # Add the top-left bounding box
        cv2.rectangle(image, rect[0], rect[1], (0, 255, 0), 2)

        # retrieve the height, width of the image
        size = image.shape[:2]

        # Calculate the bottom-right bounding box
        top = size[0] - rect[1][1]
        left = size[1] - rect[1][0]
        bottom = size[0] - rect[0][1]
        right = size[1] - rect[0][0]

        # Add the bottom-right bounding box
        cv2.rectangle(image, (left, top), (right, bottom), (0, 255, 0), 2)

        # Display the image
        cv2.imshow("card", image)

        # Wait for key press
        key = cv2.waitKey(0) & 0xFF

        # Verify whether the pressed key is q
        if key == ord('q'):
            # Destroy all cv2 windows
            cv2.destroyAllWindows()

            # Break the loop
            break


def execute(test, video_dir=None, output_dir=None, processed_dir=None):
    """
        Execute the process of extraction

        Parameters
        ----------
        test : bool
            test mode for viewing random images of cards

        Returns
        -------
    """

    # Location of the videos
    if video_dir is None:
        video_dir = "../dataset/videos_new_1/*.mov"

    # Location of the extracted images of cards
    if output_dir is None:
        output_dir = "/Volumes/Samsung_T5/datasets/test"

    # Backup folder for the videos
    if processed_dir is None:
        processed_dir = "/Volumes/Samsung_T5/datasets/processed/JC/"

    if test:
        # If test = True, display random images of cards with bounding box
        display_image(output_dir, [(5, 10), (50, 110)])
    else:
        # Capture card images from videos
        capture_images_from_video(video_dir, output_dir, processed_dir)


if __name__ == "__main__":
    # Set test mode
    test = False

    # Call execute function
    execute(test)
