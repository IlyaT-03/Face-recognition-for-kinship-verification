# Face-recognition-for-kinship-verification

This is the simplest v0.0 version of the Visual Kinship Recognition service.

### How to set up the container
`docker build -t image-similarity .`
`docker run -p 8000:8000 image-similarity`

### Working with the service
This program receives 2 images and returns the distance between them. 
It can be accessed via API:

`curl -X POST http://localhost:8000/compare -F file1=@path_to_image1.jpg -F file2=@path_to_image2.jpg`

```
import requests

url = "http://localhost:8000/compare"
files = {
    "file1": open(r"C:\path\to\image1.jpg", "rb"),
    "file2": open(r"C:\path\to\image2.jpg", "rb")
}

response = requests.post(url, files=files)
print(response.json())
```
The response should look like `{"similarity_distance": distance}`.
If no face was detected on one of the pictures, an error message is returned.

### Interpreting results
The cosine similarity the program returns is a float number between 0 and 2.
It is assumed that the lesser this number is, the more likely 2 people are relatives.

Using these distances and ground truth, you can calculate such metrics as AUC-ROC on your marked up dataset of image pairs.
It makes sense to choose the best threshold based on your dataset.
However, if you want to receive binary answers straight away, you can, for instance, use a threshold of `0.938`

Ideally, the algorithm's quality should probably be assessed with cross-validation.
