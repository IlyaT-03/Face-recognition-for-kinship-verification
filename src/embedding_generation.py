from insightface.app import FaceAnalysis
import cv2
import numpy as np
from tqdm import tqdm


def add_padding(img,
                background_height=800,
                background_width=800):
    h, w, c = img.shape
    background = np.full((background_height, background_width, 3), 255, dtype=np.uint8)
    yoff = round((background_height - h) / 2)
    xoff = round((background_width - w) / 2)
    background[yoff:yoff + h, xoff:xoff + w] = img
    return background


def get_information_from_img_path(app,
                                  img_path,
                                  resize_params=None,
                                  padding_params=None):
    image = cv2.imread(img_path)
    if image is None:
        print(f"Warning: Could not read image {img_path}")
        return None
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    if resize_params is not None:
        image = cv2.resize(image, **resize_params)

    if padding_params:
        if isinstance(padding_params, dict):
            image = add_padding(image, **padding_params)
        elif padding_params is True:
            image = add_padding(image)  # Default to 800x800

    faces = app.get(image)
    if not faces:
        return None
    return faces[0]


def get_files_faces(files,
                         resize_params=None,
                         padding_params=None,
                         model_name="buffalo_sc",
                         model_params=None):
    if model_params is None:
        model_params = {"ctx_id": 0, "det_size": (640, 640)}

    app = FaceAnalysis(name=model_name)
    app.prepare(**model_params)

    faces_dict = {}
    for file in tqdm(files):
        res = get_information_from_img_path(app, file, resize_params, padding_params)
        faces_dict[file] = res
    return faces_dict
