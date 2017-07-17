from django.views.decorators.csrf import csrf_exempt
from django.http import JsonResponse
import numpy as np
import urllib
import cv2

# define the path to the face detector
FACE_DETECTOR_PATH = "{}/opencv-3.1.0/data/haarcascades/haarcascade_frontalface_default.xml".format(
    base_path=os.path.abspath(os.path.dirname('haarcascade_frontalface_default.xml')))


@csrf_exempt
def detect(request):
    # initialize the data dictionary to be returned by the request
    data = {"success": False}

    if request.method == "POST":
        if request.FILES.get("image", None) is not None:
            image = _grab_image(stream=request.FILES["image"])

        else:
            url = request.POST.get("url")  # , None)
            print request.POST

            if url is None:
                data["error"] = "No URL provided."
                return JsonResponse(data)

            image = _grab_image(url=url)

        # convert the image to grayscale,
        # load the face cascade detector,
        # and detect faces in the image
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        detector = cv2.CascadeClassifier(FACE_DETECTOR_PATH)
        rects = detector.detectMultiScale(image, scaleFactor=1.1, minNeighbors=5,
                                          minSize=(30, 30), flags=cv2.CASCADE_SCALE_IMAGE)

        # construct a list of bounding boxes from the detection
        rects = [(int(x), int(y), int(x + w), int(y + h)) for (x, y, w, h) in rects]

        # update the data
        data.update({"num_faces": len(rects), "faces": rects, "success": True})

    return JsonResponse(data)


def _grab_image(path=None, stream=None, url=None):
    # load the image from disk
    if path is not None:
        image = cv2.imread(path)

    else:
        if url is not None:
            resp = urllib.urlopen(url)
            data = resp.read()

        elif stream is not None:
            data = stream.read()

        # convert the image to a NumPy array and then read it into OpenCV format
        image = np.asarray(bytearray(data), dtype="uint8")
        image = cv2.imdecode(image, cv2.IMREAD_COLOR)

    return image

# TODO
# 1.Make a feature rich DjangoREST framework template.
# 2.Add many other object detection support to the API.