import cv2
from PIL import Image
import onnxruntime as ort
import numpy as np
import glob
import os
import shutil
from tqdm import tqdm
import torchvision.transforms as transforms


def softmax(x):
    x = x.reshape(-1)
    e_x = np.exp(x)
    return e_x / e_x.sum(axis=0)


def postprocess(result):
    return softmax(np.array(result)).tolist()


if __name__ == "__main__":
    onnx_model_path = "ckpt0523/checkpoint-best.onnx"  # onnx模型
    ort_session = ort.InferenceSession(onnx_model_path, providers=[ 'CPUExecutionProvider'])
    # 输入层名字
    onnx_input_name = ort_session.get_inputs()[0].name
    # 输出层名字
    onnx_outputs_names = ort_session.get_outputs()[0].name

    image_dir = '/home/cw/Desktop/liu/dataset/MobileNetv3/test'
    output_dir = './0523_test'
    imgs = glob.glob(os.path.join(image_dir, '*.jpg'))
    transforms_ = [
        transforms.Resize((800, 800)),
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))  # (x-mean) / std
    ]
    transforms =transforms.Compose(transforms_)
    for img_path in tqdm(imgs):
        img_name = img_path.split('/')[-1].split('.')[0]
        img = Image.open(img_path).convert("L")  # 需要识别的图像读为灰度
        input = transforms(img)
        input = np.array(input.unsqueeze(0))
        onnx_result = ort_session.run([onnx_outputs_names], input_feed={onnx_input_name: input})
        res = postprocess(onnx_result)  # softmax
        idx = np.argmax(res)
        # print(res)
        # print(idx)  # 打印识别结果
        # print(res[idx])  # 对应的概率
        shutil.copy(img_path, os.path.join(output_dir, f'{img_name}_{idx}_{res[idx]:.2f}.jpg'))

