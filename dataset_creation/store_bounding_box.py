import numpy as np
import cv2
import glob
from tqdm import tqdm
import pickle
import json


def pre_process(image, debug):
    """
        Pre-process the corner image before finding the contours

        Parameters
        ----------
        image : array
            Corner Image
        debug: bool
            Test mode for viewing intermittent output
        Returns
        -------
        image : array
            Return the dilated image
    """

    # Convert the image to gray scale
    corner_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Detect edge using Canny Edge detector
    corner_edge = cv2.Canny(corner_gray, 30, 200)

    # Create a kernel for dilation
    kernel = np.ones((3, 3), np.uint8)

    # Dilate the edges using the kernel
    corner_edge_dialated = cv2.dilate(corner_edge, kernel, iterations=1)

    if debug:
        # Display the image if debug is enabled
        cv2.imshow("corner", corner_edge_dialated)

    # Return the dilated image
    return corner_edge_dialated


def get_contour(card, corner, card_number, debug):
    """
        Find the convex hull for the corner

        Parameters
        ----------
        card : array
            Image
        corner: array
            Corner location
        card_number : int
            Number of the Card
        debug: bool
            Test mode for viewing intermittent output
        Returns
        -------
        convex hull : array
            Return the detected convex hull
    """

    # Create a dictionary for number of expected contours for each card.
    expected_contours_map = {
        "A": 2, "2": 2, "3": 2, "4": 2, "5": 2, "6": 2, "7": 2, "8": 2, "9": 2, "10": 3, "J": 2, "Q": 2, "K": 2
    }

    # Subtract the corner from the image
    corner_img = card[corner[0][1]:corner[1][1], corner[0][0]:corner[1][0]]

    if debug:
        # Display the corner if debug is enabled
        cv2.imshow("corner", corner_img)

    # Preprocess corner image
    processed_corner_img = pre_process(corner_img, debug)

    # Find the contours ( change the line, based on open cv version )
    # _, contours, _ = cv2.findContours(processed_corner_img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contours, _ = cv2.findContours(processed_corner_img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Sort the contours
    contours = sorted(contours, key=cv2.contourArea, reverse=True)

    # If the identified number of contours are more or equal to the expected number of contours
    if len(contours) >= expected_contours_map[card_number]:

        # Get the needed contours
        contours = contours[0:expected_contours_map[card_number]]

        # Initialize the output_contours
        output_contours = None

        # count is used to keep track of number of effective contours
        # if enough contours are not available None will be returned
        count = 0

        # Loop through the sorted contours
        for contour in contours:

            # Find the area of each contour
            area = cv2.contourArea(contour)

            # If the area is more than 100 then add that to output_contours
            if area > 100:
                if output_contours is None:
                    output_contours = contour
                else:
                    output_contours = np.concatenate((output_contours, contour))

                # Increase the count of contour as its area is more than 100
                count += 1
        # If number of expected contours matches with count
        if count == expected_contours_map[card_number]:

            # Find the Convex Hull
            convex_hull = cv2.convexHull(output_contours)

            # Find the contour area of the convex hull
            convex_hull_area = cv2.contourArea(convex_hull)

            # Discard the convex hull if the area is too big
            if convex_hull_area > 3000:
                if debug:
                    # Display the image if the area of convex hull is very large
                    print(convex_hull_area)
                    cv2.drawContours(card, [convex_hull + corner[0]], 0, (0, 255, 0), 2)
                    cv2.imshow("Large Hull", card)
                    cv2.waitKey(0)
                return None
            else:
                # Move the convex hull to the correct position of the input image
                # and return it
                return convex_hull + corner[0]
        # If not enough contours is found, return None
        else:
            return None
    else:

        # If enough contours are not detected return None
        return None


def get_card_number(dir):
    """
        Find the card number from the directory

        Parameters
        ----------
        dir : string
            Directory name

        Returns
        -------
        card number : int
            Return the card number
    """

    # If the length of the dir name has 2 character [A-9, J-K]
    if len(dir.split("/")[-1]) == 2:
        number = dir.split("/")[-1][0]
    else:
        # This for 10
        number = dir.split("/")[-1][0:2]

    # Return the parsed card number
    return number


def process(input_dir, size, left_corner, output_dir, debug=False):
    """
        Find the contour locations and saves them in a pickle file as dictionary object
        <target class> -> <card image, left contour, right contour>

        Parameters
        ----------
        input_dir : string
            Location of the datasets
        size : tuple
            Size of the cards ( width, height )
        left_corner : tuple
            Location of the left corner
        output_dir : string
            Output dir
        debug: bool
            Test mode for viewing intermittent output
        Returns
        -------
        None
    """

    # Find all the directories in the input
    dirs = glob.glob(input_dir + "/*")

    # Card width and height
    width, height = size

    # Dictionary object to store all the processed images and contours
    cards_map = {}

    # Dictionary object to store the report data
    report = {}

    # Calculate the right corner bounding box
    right_corner = [(width - left_corner[1][0], height - left_corner[1][1]), (width - left_corner[0][0], height - left_corner[0][1])]

    # List of unused files
    unused_files = []

    # Loop through the dirs
    for dir in dirs:

        # Find the target class
        target_class = dir.split("/")[-1]

        # Get the list of images for each target class
        files = glob.glob(dir + "/*.png")

        # Initialize the list for each card type
        cards_map[target_class] = []

        # Counter for files not used
        not_used = 0

        # Loop through the list of files
        for index in tqdm(range(len(files)), desc=target_class):

            # Read the image using opencv
            card = cv2.imread(files[index], cv2.IMREAD_UNCHANGED)

            # Find the card number
            card_number = get_card_number(dir)

            # Find the contour of the left corner
            left_contour = get_contour(card, left_corner, card_number, debug)

            # Find the contour of the right corner
            right_contour = get_contour(card, right_corner, card_number, debug)

            # In case none of the contours are not None
            if left_contour is not None and right_contour is not None:

                if debug:
                    # Display the contour if debug is enabled
                    cv2.drawContours(card, [left_contour], 0, (0, 255, 0), 2)
                    cv2.drawContours(card, [right_contour], 0, (0, 255, 0), 2)
                    cv2.imshow("Card", card)
                    cv2.waitKey(10)

                # Convert the image format from BGRA to RGBA
                card = cv2.cvtColor(card, cv2.COLOR_BGRA2RGBA)

                # Save the image and corner contours in the dictionary object
                cards_map[target_class].append((files[index], left_contour, right_contour))
            else:
                # Increment the unused counter
                not_used += 1

                # Add the file name to the list
                unused_files.append(files[index])

        # Save the files count to the report dictionary object
        report[target_class] = {'Images Processed': len(files) - not_used, 'Skipped': not_used}

    # Dump the dictionary object in a pickle file
    pickle.dump(cards_map, open(output_dir + "cards_ref.pck", 'wb'))

    # Close if any opencv windows are opened
    cv2.destroyAllWindows()

    # Add the list of unused files to report
    report['unused_files'] = unused_files

    # Save the dictionary object in a json file
    with open(output_dir + "report.json", "w") as report_file:
        json.dump(report, report_file, indent=4)


def execute(debug):
    """
        Execute the process of extracting the contour

        Parameters
        ----------
        debug : bool
            test mode for viewing intermittent output
        Returns
        -------
        None
    """

    # Invoke the process function
    process(
        # Location of the datasets
        input_dir="/media/4TB/datasets/playing_cards/cards",

        # Size of the cards
        size=(280, 400),

        # Location of the left corner
        left_corner=[(5, 10), (50, 110)],

        # Output dir
        output_dir="/media/4TB/datasets/playing_cards/",

        # debug flag
        debug=debug)


if __name__ == "__main__":
    # Set test mode
    debug = False

    # Call execute function
    execute(debug)
