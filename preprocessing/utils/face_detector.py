import numpy as np

def detect_facenet_pytorch(detector, images, batch_size):
    faces = []
    for lb in np.arange(0, len(images), batch_size):
        imgs = [img for img in images[lb:lb+batch_size]]
        boxes, probs = detector.detect(imgs)

        for i, box in enumerate(boxes):
            if box is None:
                boxes[i] = np.asarray([])

        boxes = [box[0].astype(int).tolist() if box.ndim > 1 else box.tolist() for box in boxes]
        faces.extend(boxes)
    return faces