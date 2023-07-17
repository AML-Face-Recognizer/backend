import gc
import math
from typing import List, Tuple, Union

import cv2 as cv
import numpy as np
import torch
import torch.nn as nn
import torchvision.transforms as transforms
import face_alignment

from sklearn.metrics.pairwise import cosine_similarity
from models import inception_resnet_v1


class FaceRecognizer:
    RESCALE_SIZE = 160
    # SIMILARITY_THR = 0.471855
    SIMILARITY_THR = 0
    N_CLASSES_PRETRAIN = 500
    EMBED_DIM = 1792
    FINAL_SHAPE = (178, 218)
    FINAL_TOP_MARGIN_FROM_NOSE_BORDER = 110
    FINAL_LEFT_MARGIN_FROM_NOSE_BORDER = 90
    DIST_BETWEEN_NOSE_BRIDGE_AND_BOTTOM_LIP = 70

    def __init__(self):
        print('Loading embeddings model...')
        self.embeddings_model, self.embeddings_transform = self._load_embeddings_model()
        print('embeddings model was loaded successfully')
        self._id2embedding = []
        self._id2count = []
        self._id2name = []

    def _load_embeddings_model(self):
        """
        Load model from checkpoint (has to be downloaded into ./models/embeddings_checkpoint.pth)
        Returns: None
        """
        model = inception_resnet_v1.InceptionResnetV1(pretrained='vggface2',
                                                      classify=False,
                                                      num_classes=self.N_CLASSES_PRETRAIN)
        for param in model.parameters():
            param.requires_grad_grad = False
        model.cuda()
        model.last_linear = torch.nn.Linear(self.EMBED_DIM, self.N_CLASSES_PRETRAIN, bias=False)
        model.last_bn = torch.nn.BatchNorm1d(self.N_CLASSES_PRETRAIN, eps=0.001, momentum=0.1, affine=True)
        model.load_state_dict(torch.load('./models/embeddings_checkpoint.pth'))
        model = torch.nn.Sequential(*(list(model.children())[:-4]),
                                    nn.Flatten())
        model.eval()
        model.cuda()
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Resize(size=(self.RESCALE_SIZE, self.RESCALE_SIZE), antialias=True),
            transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
        ])
        return model, transform

    def _coords_to_deg(self, x1, y1, x2, y2):
        deltaX = x2 - x1;
        deltaY = y2 - y1;
        rad = math.atan2(deltaY, deltaX);
        deg = rad * (180 / math.pi)
        return deg

    def _rotate_image(self, image, angle, image_center=None):
        height, width = image.shape[:2]
        if image_center is None:
            cx, cy = width / 2, height / 2
            image_center = (cx, cy)
        M = cv.getRotationMatrix2D(image_center, angle, 1.0)
        result = cv.warpAffine(image, M, (width, height), flags=cv.INTER_LINEAR)
        return result

    def _get_points_position_after_rotation(self, initial_image, points, angle, image_center=None):
        height, width = initial_image.shape[:2]
        if image_center is None:
            cx, cy = width / 2, height / 2
            image_center = (cx, cy)
        M = cv.getRotationMatrix2D(image_center, angle, 1.0)
        rotated_points = cv.transform(np.array(points).reshape((-1, 1, 2)), M)
        return rotated_points.reshape((-1, 2))

    def _align_faces(self, image: np.array) -> List[np.array]:
        """
        Align faces on the image
        Args:
            image: RGB image with people faces

        Returns: list of aligned and cropped faces
        """
        initial_shape = image.shape
        initial_height, initial_width = initial_shape[:2]

        fa = face_alignment.FaceAlignment(face_alignment.LandmarksType.TWO_D, face_detector='blazeface',
                                          face_detector_kwargs={'back_model': True})
        preds = fa.get_landmarks_from_image(image, return_landmark_score=True)

        if preds[0] is None:
            raise Exception('There should be at least one face on the image')

        image_with_margin = cv.copyMakeBorder(image, top=initial_shape[0], bottom=initial_shape[0],
                                              left=initial_shape[1], right=initial_shape[1],
                                              borderType=cv.BORDER_REPLICATE)

        landmarks_list = preds[0]

        faces_aligned = []

        for landmarks in landmarks_list:
            side_left_coords = landmarks[0]
            side_right_coords = landmarks[16]
            main_pivot = landmarks[27]

            deg_side = self._coords_to_deg(*side_left_coords, *side_right_coords)

            landmarks_with_margin = [(landmark[0] + initial_width, landmark[1] + initial_height) for landmark in
                                     landmarks]

            image_rotated = self._rotate_image(image_with_margin, deg_side)

            landmarks_rotated = self._get_points_position_after_rotation(image_with_margin, landmarks_with_margin,
                                                                         deg_side)

            image_rotated = image_rotated[initial_height:initial_height * 2, initial_width:initial_width * 2]

            landmarks_rotated = [(landmark[0] - initial_width, landmark[1] - initial_height) for landmark in
                                 landmarks_rotated]

            main_pivot = landmarks_rotated[27]
            bottom_pivot = landmarks_rotated[8]
            # left_pivot = landmarks_rotated[36]

            ratio = (bottom_pivot[1] - main_pivot[1]) / self.DIST_BETWEEN_NOSE_BRIDGE_AND_BOTTOM_LIP

            left_top_corner = (int(main_pivot[0] - self.FINAL_LEFT_MARGIN_FROM_NOSE_BORDER * ratio),
                               int(main_pivot[1] - self.FINAL_TOP_MARGIN_FROM_NOSE_BORDER * ratio))
            right_bottom_corner = (
                int(main_pivot[0] + (self.FINAL_SHAPE[0] - self.FINAL_LEFT_MARGIN_FROM_NOSE_BORDER) * ratio),
                int(main_pivot[1] + (self.FINAL_SHAPE[1] - self.FINAL_TOP_MARGIN_FROM_NOSE_BORDER) * ratio))

            top_padding = -left_top_corner[1]
            if top_padding < 0: top_padding = 0
            left_padding = 0 - left_top_corner[0]
            if left_padding < 0: left_padding = 0

            right_padding = right_bottom_corner[0] - initial_shape[0]
            if right_padding < 0: right_padding = 0
            bottom_padding = right_bottom_corner[1] - initial_shape[1]
            if bottom_padding < 0: bottom_padding = 0

            left_top_corner = (left_top_corner[0], top_padding + left_top_corner[1])
            right_bottom_corner = (right_bottom_corner[0], top_padding + right_bottom_corner[1])
            left_top_corner = (left_padding + left_top_corner[0], left_top_corner[1])
            right_bottom_corner = (left_padding + right_bottom_corner[0], right_bottom_corner[1])

            final_image = cv.copyMakeBorder(image_rotated, top=top_padding, bottom=bottom_padding, left=left_padding,
                                            right=right_padding, borderType=cv.BORDER_REPLICATE)

            final_image = final_image[left_top_corner[1]:right_bottom_corner[1],
                          left_top_corner[0]:right_bottom_corner[0]]

            final_image = cv.resize(final_image, self.FINAL_SHAPE)

            faces_aligned.append(final_image)

        return faces_aligned

    def _get_face_embedding(self, aligned: Union[np.array, List[np.array]]) -> np.array:
        """
        Given the aligned face photo, convert into latent representation
        Args:
            aligned: RGB image of shape (self.RESCALE_SIZE, self.RESCALE_SIZE) or list of such images

        Returns: vector of shape (self.EMBED_DIM, ) or matrix of shape (n_images, self.EMBED_DIM),
         where n_images = # of images iin aligned

        """

        if isinstance(aligned, np.ndarray):
            inp = self.embeddings_transform(aligned).unsqueeze(0)
        else:
            inp = torch.stack([self.embeddings_transform(face) for face in aligned])
        embedding = self.embeddings_model(inp.cuda()).detach().cpu().numpy()
        return embedding.squeeze() if isinstance(aligned, np.ndarray) else embedding

    def _align_one_face(self, face_photo: np.array) -> np.array:
        """
        Args:
            face_photo: photo of a single human face

        Returns: aligned face

        """
        aligned = self._align_faces(face_photo)

        aligned_face = aligned[0]
        if len(aligned) != 1:
            return None
        return aligned_face

    def add_person(self, face_photo: np.array, name: str) -> int:
        """
        Store embedding for the new person
        Args:
            face_photo: RGB photo with single human on it
            name: name, under which the person is stored

        Returns: -1 if the operation has failed, person's id otherwise

        """
        aligned_face = self._align_one_face(face_photo)
        if aligned_face is None:
            print("Error: no or multiple faces were detected when called FaceRecognizer.add_person")
            return -1
        face_embedding = self._get_face_embedding(aligned_face)
        person_id = len(self._id2embedding)
        self._id2embedding.append(face_embedding)
        self._id2count.append(1)
        self._id2name.append(name)
        torch.cuda.empty_cache()
        gc.collect()
        return person_id

    def add_photo(self, person_id: int, face_photo: np.array) -> bool:
        """
        Given person's new photo and name, modify the stored embedding with the new photo (running mean is used)
        Args:
            person_id: person's id (retured by self.add_photo)
            face_photo: RGB image

        Returns: False if face was not found or person has not been previously added, True is embedding was modified
        """
        if person_id >= len(self._id2embedding):
            return False
        aligned_face = self._align_one_face(face_photo)
        if aligned_face is None:
            print("Error: no or multiple faces were detected when called FaceRecognizer.add_photo")
            return False
        face_embedding = self._get_face_embedding(aligned_face)
        N = self._id2count[person_id]
        self._id2embedding[person_id] = (self._id2embedding[person_id] * N + face_embedding) / (N + 1)
        self._id2count[person_id] += 1
        return True

    def recognize(self, image: np.array) -> List[Tuple[np.array, str, int, float]]:
        """
        Recognize already seen (through self.add_person and self.add_photo) people on image
        Args:
            image: RGB image

        Returns: For each of the faces found during alignment, return: aligned face, name, id and confidence
        """
        aligned_faces = self._align_faces(image)
        if len(aligned_faces) == 0 or len(self._id2count) == 0:
            return None
        faces_embeddings = self._get_face_embedding(aligned_faces)
        similarities = cosine_similarity(np.array(self._id2embedding), faces_embeddings)
        labels, conf = np.argmax(similarities, axis=0), np.max(similarities, axis=0)
        return [(aligned_faces[i], self._id2name[labels[i]], labels[i], conf[i]) for i in range(len(labels)) if
                conf[i] >= self.SIMILARITY_THR]


# Sample usage below
if __name__ == "__main__":
    recognizer = FaceRecognizer()
    from PIL import Image

    face_id = recognizer.add_person(np.array(Image.open('../face-smiling.jpg')), 'Face')
    recognizer.add_photo(face_id, np.array(Image.open('../face-mic.webp')))
    basta_id = recognizer.add_person(np.array(Image.open('../basta.jpg')), 'Basta')
    result = recognizer.recognize(np.array(Image.open('../face-old.jpg')))
    print(result[0][1:])  # ('Mary', 0, 0.8681366)
