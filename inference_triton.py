import os
import random
import numpy as np
import onnxruntime as ort
import numpy as np
import cv2
from enum import Enum
import threading
import tritongrpcclient


triton_client = tritongrpcclient.InferenceServerClient(url='localhost:8001')


def pad_img(image, size: int):
    h, w, c = image.shape
    dw = (size-w)//2
    dh = (size-h)//2
    image = cv2.copyMakeBorder(image, dh, dh+h %
                             2, dw, dw+w % 2, borderType=cv2.BORDER_REFLECT)
    return image.transpose((2, 0, 1))[np.newaxis]


def process_ensemble(rgb_image: np.array):  # assuming images are around 128x128
    result_values = []
    done_event = threading.Event()

    def callback(result, error):
        result_values.append(result)
        if len(result_values) == 2:  # 1 inference for rine, 2 inference for resnet
            done_event.set()

    result = []

    img = pad_img(rgb_image, 224)
    outputs = [tritongrpcclient.InferRequestedOutput('probas_bin')]
    inputs = [tritongrpcclient.InferInput('input', (1, 3, 224, 224), "UINT8")]
    inputs[0].set_data_from_numpy(img)
    req = triton_client.async_infer(
        model_name='rine_vit', inputs=inputs, outputs=outputs, callback=callback)
    result.append(req)

    # resnet
    img = pad_img(rgb_image, 256)
    img2 = np.flip(img, axis=3)
    resnet_batch = np.concatenate([img, img2], axis=0)
    outputs = [tritongrpcclient.InferRequestedOutput('probas_bin'),
               tritongrpcclient.InferRequestedOutput('probas_multi')]
    inputs = [tritongrpcclient.InferInput('input', (2, 3, 256, 256), "UINT8")]
    inputs[0].set_data_from_numpy(resnet_batch)
    req = triton_client.async_infer(
        model_name='resnet', inputs=inputs, outputs=outputs, callback=callback)
    result.append(req)

    done_event.wait()

    probas = []
    # print(len(result_values))
    for req in result_values:
        probs_bin = req.as_numpy('probas_bin')[:, 1]
        if probs_bin.shape[0] == 2:
            probas.append((probs_bin[0]+probs_bin[1])/2.0)
            probas_multi = req.as_numpy('probas_multi')
        else:
            probas.append(probs_bin[0])

    return (probas[0]+probas[1])/2.0, np.median(probas_multi, axis=0)


def read_img(img_path):
    image = cv2.imread(img_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = image[0:1024, 0:1024]
    image = cv2.resize(image, (128, 128))
    return image


if __name__ == '__main__':

    import sys
    img_path = sys.argv[1]
    image = read_img(img_path)
    out = process_ensemble(image)

    print('ensemble result', out)
