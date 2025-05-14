import os
import random
import numpy as np
import onnxruntime as ort
import numpy as np
import cv2
from enum import Enum
import sys


def load_onnx_model(model_path):
    """Load ONNX model from path, using best available provider"""
    providers = ["CUDAExecutionProvider", "CPUExecutionProvider"]
    so = ort.SessionOptions()
    so.log_severity_level = 3

    return ort.InferenceSession(model_path, providers=providers, sess_options=so)


def run_inference(model, image):
    """Run inference on preprocessed image"""
    outputs = model.get_outputs()
    input_name = model.get_inputs()[0].name
    output_names = [k.name for k in outputs]
    result = model.run(output_names, {input_name: image})

    return result


class EnsembleSmallModel:
    def __init__(self):
        self.model1 = load_onnx_model('resnet/1/model.onnx')
        self.model2 = load_onnx_model('rine_vit/1/model.onnx')

    def run_inference(self, rgb_img: np.array):
        img1 = pad_img(image=rgb_img.copy(), size=256)
        img2 = np.flip(img1, axis=3)
        resnet_batch = np.concatenate([img1, img2], axis=0)

        rine_batch = pad_img(image=rgb_img.copy(), size=224)
        resnet_pred = run_inference(model=self.model1, image=resnet_batch)
        rine_pred = run_inference(model=self.model2, image=rine_batch)

        ai_prob = (resnet_pred[0][:, 1].mean() + rine_pred[0][:, 1])/2.0
        model_probs = resnet_pred[1].mean(axis=0)
        return ai_prob, model_probs


def pad_img(image: np.array, size: int):
    h, w, c = image.shape
    dw = (size-w)//2
    dh = (size-h)//2
    image = cv2.copyMakeBorder(image, dh, dh+h %
                               2, dw, dw+w % 2, borderType=cv2.BORDER_REFLECT)
    image = image.transpose((2, 0, 1))[np.newaxis]
    return image


def read_img(img_path):
    img_ext = img_path.split('.')[-1]
    img = cv2.imread(img_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = img[0:1024, 0:1024]  # emulating thumbnail
    img = cv2.resize(img, (128, 128))
    if img_ext == 'png':
        img = cv2.imdecode(cv2.imencode('.jpg', img, [cv2.IMWRITE_JPEG_QUALITY, 99])[
                           1], cv2.IMREAD_COLOR)

    return img


if __name__ == '__main__':
    img_path = sys.argv[1]

    image = read_img(img_path)

    import sys
    clf = EnsembleSmallModel()
    out = clf.run_inference(rgb_img=image)
    print(out)
