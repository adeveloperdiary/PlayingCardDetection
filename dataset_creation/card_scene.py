import numpy as np
import cv2
import imgaug
from dataset_creation.scene_objects import Objects, Cards, Backgrounds
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from shapely.geometry import Polygon
import os

xml_body_1 = """<annotation>
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
xml_object = """ 
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
xml_body_2 = """</annotation>        
"""


class BBA:  # Bounding box + annotations
    def __init__(self, bb, classname):
        self.x1 = int(round(bb.x1))
        self.y1 = int(round(bb.y1))
        self.x2 = int(round(bb.x2))
        self.y2 = int(round(bb.y2))
        self.classname = classname


class Scene:
    def __init__(self,
                 object_instance: Objects,
                 background_instance: Objects,
                 scene_dim={},
                 object_dim={}, overlap_ratio=0):
        self.object_instance = object_instance
        self.background_instance = background_instance
        self.scene_width = scene_dim['width']
        self.scene_height = scene_dim['height']
        self.object_width = object_dim['width']
        self.object_height = object_dim['height']

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

        # imgaug transformation for one card in scenario with 2 cards
        self.simple_transformations = imgaug.augmenters.Sequential([
            imgaug.augmenters.Affine(scale=[0.65, 1]),
            imgaug.augmenters.Affine(rotate=(-180, 180)),
            imgaug.augmenters.Affine(translate_percent={"x": (-0.25, 0.25), "y": (-0.25, 0.25)}),
        ])

        # imgaug transformation for the background
        self.scale_bg = imgaug.augmenters.Scale({"height": self.scene_height, "width": self.scene_width})

        self.bounding_boxes = []

    def key_points_2_polygon(self, kps):
        """
            Convert imgaug keypoints to shapely polygon
        """
        pts = [(kp.x, kp.y) for kp in kps]
        return Polygon(pts)

    def key_points_2_bounding_box(self, kps):
        """
            Determine imgaug bounding box from imgaug keypoints
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

    def __augment(self, img, hulls_key_points, restart=True):
        """
            Apply augmentation 'seq' to image 'img' and keypoints 'list_kps'
            If restart is False, the augmentation has been made deterministic outside the function (used for 3 cards scenario)
        """
        # Make sequence deterministic
        while True:
            if restart:
                myseq = self.simple_transformations.to_deterministic()
            else:
                myseq = self.simple_transformations
            # Augment image, keypoints and bbs
            img_aug = myseq.augment_images([img])[0]

            # Add the Card Key Point tp the hulls
            list_kps = [self.card_key_point] + hulls_key_points

            list_kps_aug = [myseq.augment_keypoints([kp])[0] for kp in list_kps]

            list_bbs = [self.key_points_2_bounding_box(key_points) for key_points in list_kps_aug[1:]]

            valid = True
            # Check the card bounding box stays inside the image
            for bb in list_bbs:
                if bb is None or int(round(bb.x2)) >= self.scene_width or int(round(bb.y2)) >= self.scene_height or int(bb.x1) <= 0 or int(
                        bb.y1) <= 0:
                    valid = False
                    break
            if valid:
                break
            elif not restart:
                img_aug = None
                break

        return img_aug, list_kps_aug, list_bbs

    def __hulls_to_key_points(self, hulls=[]):
        key_points = []

        for hull in hulls:
            # hull is a cv2.Contour, shape : Nx1x2
            key_point = [imgaug.Keypoint(x=p[0] + self.centerX, y=p[1] + self.centerY) for p in hull.reshape(-1, 2)]
            key_point = imgaug.KeypointsOnImage(key_point, shape=(self.scene_height, self.scene_width, 3))
            key_points.append(key_point)
        return key_points

    def create_default_scene(self, num_obj=0):
        self.bounding_boxes = []
        self.final_background = self.scale_bg.augment_image(self.background_instance.get_random())
        all_key_points = []

        bounding_boxes_dict = {}

        for num in range(num_obj):

            obj_img, obj_name, hulls = self.object_instance.get_random()
            hulls_key_points = self.__hulls_to_key_points(hulls)

            image = np.zeros((self.scene_height, self.scene_width, 4), dtype=np.uint8)
            image[self.centerY:self.centerY + self.object_height, self.centerX:self.centerX + self.object_width, :] = obj_img
            image, key_points, bounding_boxes = self.__augment(image, hulls_key_points)

            card_rect = self.key_points_2_polygon(key_points[0].keypoints[0:4])

            if len(all_key_points) > 0:
                for key_point in all_key_points:
                    key_point_rect = self.key_points_2_polygon(key_point.keypoints[:])
                    intersect = card_rect.intersection(key_point_rect)
                    if intersect.area > self.overlap_ratio:
                        # Remove Bounding Box
                        bounding_box = bounding_boxes_dict[key_point.__hash__()]
                        if bounding_box in self.bounding_boxes:
                            self.bounding_boxes.remove(bounding_box)

            bba_bounding_box = []

            for bb in bounding_boxes:
                bba = BBA(bb, obj_name)
                self.bounding_boxes.append(bba)
                bba_bounding_box.append(bba)

            for key_point, b_box in zip(key_points[1:], bba_bounding_box):
                all_key_points.append(key_point)
                bounding_boxes_dict[key_point.__hash__()] = b_box

            # For all Channel
            image_mask = np.stack([image[:, :, 3]] * 3, -1)
            # Overlap Image
            self.final_background = np.where(image_mask, image[:, :, 0:3], self.final_background)

    def display(self):
        fig, ax = plt.subplots(1, figsize=(8, 8))
        ax.imshow(self.final_background)
        for bb in self.bounding_boxes:
            rect = patches.Rectangle((bb.x1, bb.y1), bb.x2 - bb.x1, bb.y2 - bb.y1, linewidth=1, edgecolor='b', facecolor='none')
            ax.add_patch(rect)
        plt.show()

    def get_file_names(self, save_dir):

        from hashlib import md5
        from time import localtime
        import random
        bname = "%09d" % random.randint(0, 999999999)

        r = md5(str(localtime()).encode('utf-8')).hexdigest()+bname
        return os.path.join(save_dir, "%s_%s" % (r, 'image.jpg')), os.path.join(save_dir, "%s_%s" % (r, 'objects.xml'))

    def write_files(self, save_dir, add_bb=False):
        image_file, xml_file = self.get_file_names(save_dir)
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
        self.create_voc_xml(xml_file, image_file)

    def create_voc_xml(self, xml_file, image_file):
        with open(xml_file, "w") as f:
            f.write(xml_body_1.format(
                **{'FILENAME': os.path.basename(image_file), 'PATH': image_file, 'WIDTH': self.scene_width, 'HEIGHT': self.scene_height}))
            for bba in self.bounding_boxes:
                f.write(xml_object.format(**{'CLASS': bba.classname, 'XMIN': bba.x1, 'YMIN': bba.y1, 'XMAX': bba.x2, 'YMAX': bba.y2}))
            f.write(xml_body_2)


from tqdm import tqdm

if __name__ == '__main__':
    cards = Cards(path="/Volumes/Samsung_T5/datasets/cards_ref.pck")
    bg = Backgrounds(path="/Volumes/Samsung_T5/datasets/dtd/images/*/*.jpg")
    scene = Scene(cards, bg, scene_dim={'width': 720, 'height': 720}, object_dim={'width': 280, 'height': 400}, overlap_ratio=400)

    for count in tqdm(range(100)):
        scene.create_default_scene(num_obj=count % 4 + 1)
        #scene.display()
        scene.write_files(save_dir="/Volumes/Samsung_T5/datasets/final", add_bb=True)
