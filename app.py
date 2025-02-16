from fastapi import FastAPI, File, UploadFile
import cv2
import io
from PIL import Image
import numpy as np
from scipy.spatial.distance import cosine
from insightface.app import FaceAnalysis

app = FastAPI()

model_app = FaceAnalysis(name="buffalo_sc")
model_app.prepare(ctx_id=0, det_size=(640,640))


async def read_image(file: UploadFile):
    contents = await file.read()  # Read file bytes
    pil_image = Image.open(io.BytesIO(contents))  # Convert bytes to Pillow Image
    opencv_image = np.array(pil_image)  # Convert to NumPy
    return cv2.cvtColor(opencv_image, cv2.COLOR_RGB2BGR)  # Convert RGB to BGR


def get_face(image):
    faces = model_app.get(image)
    if len(faces) == 0:
        return None
    return faces[0]

@app.post("/compare")
async def compare_images(file1: UploadFile = File(...), file2: UploadFile = File(...)):
    image1 = await read_image(file1)
    image2 = await read_image(file2)
    face1 = get_face(image1)["embedding"]
    face2 = get_face(image2)["embedding"]
    if face1 is None or face2 is None:
        no_faces = []
        if face1 is None:
            no_faces.append(file1)
        if face2 is None:
            no_faces.append(file2)
        return {"error": f"face not found in images {no_faces}"}

    distance = cosine(face1, face2)
    return {"similarity_distance": distance.item()}

