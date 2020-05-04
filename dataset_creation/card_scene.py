import numpy as np
import imgaug
from dataset_creation.scene_objects import Objects, Cards, Backgrounds
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from shapely.geometry import Polygon
import os


class VOCXML:
    """
        An implementation class for storing the object
        annotation using VOC XML format for YOLO.

        Methods
        -------
        __init__(bb,classname) [constructor]
            - initializes the class
        create_voc_xml(xml_file, image_file, bounding_boxes_with_annotation) [public]
            - saves the xml file

    """

    def __init__(self):
        """
            Constructor for the VOCXML class.
                - Saves the template of the VOC XML
        """
        self.xml_body_1 = """<annotation>
                <folder>FOLDER</folder>
                <filename>{FILENAME}</filename>
                <path>{PATH}</path>
                <source>
                        <database>Unknown</database>
                </source>
                <size>
                        <width>{WIDTH}</width>
                        <height>{HEIGHT}</height>
                        <depth>3</depth>
                </size>
        """
        self.xml_object = """ 
                <object>
                        <name>{CLASS}</name>
                        <pose>Unspecified</pose>
                        <truncated>0</truncated>
                        <difficult>0</difficult>
                        <bndbox>
                                <xmin>{XMIN}</xmin>
                                <ymin>{YMIN}</ymin>
                                <xmax>{XMAX}</xmax>
                                <ymax>{YMAX}</ymax>
                        </bndbox>
                </object>
        """
        self.xml_body_2 = """</annotation>        
        """

    def create_voc_xml(self, xml_file, image_file, bounding_boxes_with_annotation, scene_dim={}):
        """
            Saves the annotation details in VOC xml format.
                - Using the template constructs the xml
                - loop through the bounding boxes and saves that information
                - write details to the desk

            :parameter
            ----------
            xml_file : string
                - the xml file full path
            image_file : string
                - the full path of the image
            bounding_boxes_with_annotation : list
                - list of bounding boxes and annotation
            scene_dim : dict
                - The height and width of the scene.

        """
        with open(xml_file, "w") as f:
            f.write(self.xml_body_1.format(
                **{'FILENAME': os.path.basename(image_file), 'PATH': image_file, 'WIDTH': scene_dim['width'], 'HEIGHT': scene_dim['height']}))
            for bba in bounding_boxes_with_annotation:
                f.write(self.xml_object.format(**{'CLASS': bba.classname, 'XMIN': bba.x1, 'YMIN': bba.y1, 'XMAX': bba.x2, 'YMAX': bba.y2}))
            f.write(self.xml_body_2)


class BoundingBoxWithAnnotation:
    """
        An implementation class for combining
        Bounding Box and Class Annotation

        Methods
        -------
        __init__(bb,classname) [public]

    """

    def __init__(self, bb, classname):
        """
            Constructor for the BoundingBoxWithAnnotation class.

            :parameter
            ----------
            bb : imgaug.BoundingBox
               - Bounding box for the key point
            classname : string
               - Name of the object
        """
        self.x1 = int(round(bb.x1))
        self.y1 = int(round(bb.y1))
        self.x2 = int(round(bb.x2))
        self.y2 = int(round(bb.y2))
        self.classname = classname


class Scene:
    """
        This class is responsible for creating the scene using one background and
        multiple card objects.
            - The class is written such way that it is reusable for any type of
              backgrounds and object types.

        Methods
        -------
        __init__ : constructor
            - initializes the class using the arguments
        create_default_scene : public
            - this is the primary method which generates the scene
        _key_points_2_polygon : private
            - Utility method, used to convert the key points to polygon objects
        _key_points_2_bounding_box : private
            - Utility method, used to convert the key points to bounding box
        _hulls_to_key_points : private
            - Utility method, used to convert the hulls to key points
        _get_file_names : private
            - Utility method, used to generate random unique file names
        _augment : private
            - Transform the object and hulls
        write_files : public
            - save scene and xml
        display : public
            - show the generated scene
    """

    def __init__(self,
                 object_instance: Objects,
                 background_instance: Objects,
                 scene_dim={},
                 object_dim={}, overlap_ratio=0):
        """
            Initialization for Scene class

            :parameter
            ----------
            object_instance : Objects
                - Instance of object, used to generate object randomly.
            background_instance : Objects
                - Instance of the background object
            scene_dim : dict[width,height]
                - Dimension of the scene
            object_dim : dict[width,height]
                - Dimension of the object
            overlap_ratio : int
                - Parameter to set the area overlapping value.Need to adjusted accordingly.

        """
        self.object_instance = object_instance
        self.background_instance = background_instance

        self.scene_width = scene_dim['width']
        self.scene_height = scene_dim['height']
        self.object_width = object_dim['width']
        self.object_height = object_dim['height']

        # Get the center of scene based on the object size.
        # We are  centering so that after transforming the object
        # remains in the scene
        self.centerX = int((self.scene_width - self.object_width) / 2)
        self.centerY = int((self.scene_height - self.object_height) / 2)

        self.overlap_ratio = overlap_ratio

        # imgaug keypoints of the bounding box of a whole card
        self.card_key_point = imgaug.KeypointsOnImage([
            imgaug.Keypoint(x=self.centerX, y=self.centerY),
            imgaug.Keypoint(x=self.centerX + self.object_width, y=self.centerY),
            imgaug.Keypoint(x=self.centerX + self.object_width, y=self.centerY + self.object_height),
            imgaug.Keypoint(x=self.centerX, y=self.centerY + self.object_height)
        ], shape=(self.scene_height, self.scene_width, 3))

        # imgaug transformation for one card
        self.simple_transformations = imgaug.augmenters.Sequential([
            imgaug.augmenters.Affine(scale=[0.65, 1]),
            imgaug.augmenters.Affine(rotate=(-180, 180)),
            imgaug.augmenters.Affine(translate_percent={"x": (-0.25, 0.25), "y": (-0.25, 0.25)}),
        ])

        # imgaug transformation for the background
        self.scale_bg = imgaug.augmenters.Scale({"height": self.scene_height, "width": self.scene_width})

        # list to store all the final bounding boxes
        self.bounding_boxes = []

    def _key_points_2_polygon(self, kps):
        """
            Convert imgaug keypoints to shapely polygon
                - Convert the key points to Polygon object so that
                  we can calculate the overlapping key points

            :parameter
            ----------
            kps : list
                - key point of the object

            :returns
            -------
            Polygon : shapely.geometry.Polygon

        """

        pts = [(kp.x, kp.y) for kp in kps]

        return Polygon(pts)

    def _key_points_2_bounding_box(self, kps):
        """
            Determine imgaug bounding box from imgaug keypoints
                - Convert the key points to Bounding box

            :parameter
            ----------
            kps : list
                - key point of the object

            :returns
            -------
            BoundingBox : imgaug.BoundingBox
        """
        extend = 3  # To make the bounding box a little bit bigger
        kpsx = [kp.x for kp in kps.keypoints]
        minx = max(0, int(min(kpsx) - extend))
        maxx = min(self.scene_width, int(max(kpsx) + extend))
        kpsy = [kp.y for kp in kps.keypoints]
        miny = max(0, int(min(kpsy) - extend))
        maxy = min(self.scene_height, int(max(kpsy) + extend))
        if minx == maxx or miny == maxy:
            return None
        else:
            return imgaug.BoundingBox(x1=minx, y1=miny, x2=maxx, y2=maxy)

    def _augment(self, img, hulls_key_points):
        """
            Apply augmentation 'seq' to image 'img' and key points 'list_kps'

            :parameter
            ----------
            img : ndarray
                - The object
            hulls_key_points : list
                - list of convex hulls

            :returns
            -------
            img_aug : ndarray
                - Augmented Object Image
            list_kps_aug : list
                - list of augmented key points
            list_bbs : list
                - list of bounding boxes

        """
        # Make sequence deterministic
        myseq = self.simple_transformations.to_deterministic()

        # Augment the image
        img_aug = myseq.augment_images([img])[0]

        # Add the Card Key Point tp the hulls
        list_kps = [self.card_key_point] + hulls_key_points

        # Augment the keypoints
        list_kps_aug = [myseq.augment_keypoints([kp])[0] for kp in list_kps]

        # Convert the key points to bounding box.
        # Do not need the bounding box of the object
        list_bbs = [self._key_points_2_bounding_box(key_points) for key_points in list_kps_aug[1:]]

        return img_aug, list_kps_aug, list_bbs

    def _hulls_to_key_points(self, hulls=[]):
        """
            Convert the convex hulls to the key points

            :parameter
            ----------
            hulls : list
                - list of hulls

            :returns
            -------
            key_points : list
                - list of key points

        """
        key_points = []

        for hull in hulls:
            # hull is a cv2.Contour, shape : Nx1x2
            key_point = [imgaug.Keypoint(x=p[0] + self.centerX, y=p[1] + self.centerY) for p in hull.reshape(-1, 2)]
            key_point = imgaug.KeypointsOnImage(key_point, shape=(self.scene_height, self.scene_width, 3))
            key_points.append(key_point)
        return key_points

    def create_default_scene(self, num_obj=0):
        """
            Generate scene by planing objects into the background image.
                - Number of objects can be controlled using the num_obj argument

            :parameter
            num_obj : int
                - Number of objects to be placed into the scene

        """

        # reset the selected bounding_boxes
        self.bounding_boxes = []

        # Get a background randomly
        self.final_background = self.scale_bg.augment_image(self.background_instance.get_random())

        # Create a list to store all the key points
        all_key_points = []

        # Store the bounding boxes for the corresponding key point
        bounding_boxes_dict = {}

        # Loop through the num_obj to get a new object and transform it
        for _ in range(num_obj):

            # Get a random object
            obj_img, obj_name, hulls = self.object_instance.get_random()

            # Convert the hulls to the key points
            hulls_key_points = self._hulls_to_key_points(hulls)

            # Create an empty image
            image = np.zeros((self.scene_height, self.scene_width, 4), dtype=np.uint8)

            # Place the object at the center of the image
            image[self.centerY:self.centerY + self.object_height, self.centerX:self.centerX + self.object_width, :] = obj_img

            # Augment the image and key points
            image, key_points, bounding_boxes = self._augment(image, hulls_key_points)

            # before we add the object and key points to the scene,
            # determine how many previous key points are under the current
            # object so that we can remove them. This way overlapped bounding boxes
            # will not be considered.

            # Convert the key point of the entire current object [not the hulls] to Polygon,
            # as we need to determine overlapping key points between the object itself and previous
            # key points.
            card_rect = self._key_points_2_polygon(key_points[0].keypoints[0:4])

            # Skip the validation for the first object as there won't be
            # any overlapping key points.
            if len(all_key_points) > 0:
                # Loop through all the key points
                for key_point in all_key_points:
                    # Convert the key point to Polygon
                    key_point_rect = self._key_points_2_polygon(key_point.keypoints[:])
                    # Find the intersection between the object and existing key points.
                    intersect = card_rect.intersection(key_point_rect)

                    # If the intersect area is larger than overlap_ratio then remove the bounding box from the
                    # final list as this seems to be overlapped by current object.
                    if intersect.area > self.overlap_ratio:
                        # Get the bounding box for the given key points
                        # using the hash() key of the key point object from the dict
                        bounding_box = bounding_boxes_dict[key_point.__hash__()]

                        if bounding_box in self.bounding_boxes:
                            # Remove Bounding Box
                            self.bounding_boxes.remove(bounding_box)

            # Loop through the current object's key points and corresponding bounding boxes
            for key_point, b_box in zip(key_points[1:], bounding_boxes):
                # Add the object name to the bounding box
                bba = BoundingBoxWithAnnotation(b_box, obj_name)

                #  Add that to the selected bounding_boxes list
                self.bounding_boxes.append(bba)

                # Add the same to the all_key_points list
                all_key_points.append(key_point)

                # Store the bounding_boxes to the dict using hash of the key point
                bounding_boxes_dict[key_point.__hash__()] = bba

            # Duplicate the image mask for all Channels
            image_mask = np.stack([image[:, :, 3]] * 3, -1)

            # Overlap the object to the scene using the mask
            self.final_background = np.where(image_mask, image[:, :, 0:3], self.final_background)

    def display(self):
        """
            Display the scene with the bounding boxes
                - loop through the bounding boxes, then add them to the image
        """
        fig, ax = plt.subplots(1, figsize=(8, 8))
        ax.imshow(self.final_background)
        for bb in self.bounding_boxes:
            rect = patches.Rectangle((bb.x1, bb.y1), bb.x2 - bb.x1, bb.y2 - bb.y1, linewidth=1, edgecolor='b', facecolor='none')
            ax.add_patch(rect)
        plt.show()

    def _get_file_names(self, save_dir):
        """
            Generate random file names for jpg and xml

            :parameter
            save_dir : string
                - The dir where where the files needs to be stored.

            :returns
            jpg path: string
                - Path of the jpg file
            xmp path: string
                - Path of the xml file

        """

        from hashlib import md5
        from time import localtime
        import random
        bname = "%09d" % random.randint(0, 999999999)

        r = md5(str(localtime()).encode('utf-8')).hexdigest() + bname
        return os.path.join(save_dir, "%s_%s" % (r, 'objects.jpg')), os.path.join(save_dir, "%s_%s" % (r, 'objects.xml'))

    def write_files(self, save_dir, add_bb=False):
        """
            Saves the files to the fs

            :parameter
            save_dir : string
                - The dir where where the files needs to be stored.
            add_bb : bool
                - Output scene will also contain the bounding boxes.

        """
        image_file, xml_file = self._get_file_names(save_dir)
        if add_bb:
            fig, ax = plt.subplots(1, figsize=(8, 8))
            ax.imshow(self.final_background)
            for bb in self.bounding_boxes:
                rect = patches.Rectangle((bb.x1, bb.y1), bb.x2 - bb.x1, bb.y2 - bb.y1, linewidth=1, edgecolor='b', facecolor='none')
                ax.add_patch(rect)
            fig.savefig(image_file)
            plt.cla()
            plt.close(fig)
        else:
            plt.imsave(image_file, self.final_background)

        # Create an instance of the VOCXML class
        voc_xml = VOCXML()
        voc_xml.create_voc_xml(xml_file, image_file, self.bounding_boxes, {'width': self.scene_width, 'height': self.scene_height})


from tqdm import tqdm

if __name__ == '__main__':

    """
        This is used to generate the scene using the the cards and the background image.
    """

    cards = Cards(path="/media/4TB/datasets/playing_cards/cards_ref.pck")
    bg = Backgrounds(path="/media/4TB/datasets/backgrounds/images/*/*.jpg")
    # cards = Cards(path="/Volumes/Samsung_T5/datasets/cards_ref.pck")
    # bg = Backgrounds(path="/Volumes/Samsung_T5/datasets/dtd/images/*/*.jpg")
    scene = Scene(cards, bg, scene_dim={'width': 720, 'height': 720}, object_dim={'width': 280, 'height': 400}, overlap_ratio=400)

    for count in tqdm(range(20)):
        scene.create_default_scene(num_obj=count % 10 + 1)
        # scene.display()
        scene.write_files(save_dir="/media/4TB/datasets/playing_cards/scenes/test", add_bb=False)
        # scene.write_files(save_dir="/Volumes/Samsung_T5/datasets/final", add_bb=False)
