

import torch
from torchvision.models import mobilenet_v2

from onnx_tf.backend import prepare
import onnx

import tensorflow as tf

def torch_to_onnx(onnx_model_path):

    img_size = (640, 640)
    batch_size = 1
    #onnx_model_path = 'model.onnx'
    
    model = mobilenet_v2()
    model.eval()
    
    sample_input = torch.rand((batch_size, 3, *img_size))
    
    y = model(sample_input)
    
    torch.onnx.export(
        model,
        sample_input, 
        onnx_model_path,
        verbose=False,
        input_names=['input'],
        output_names=['output'],
        opset_version=12
    )


def onnx_to_tf(onnx_model_path, tf_model_path):
    onnx_model = onnx.load(onnx_model_path)
    tf_rep = prepare(onnx_model)
    tf_rep.export_graph(tf_model_path)
    


def tf_to_tflite(saved_model_dir, tflite_model_path):
    converter = tf.lite.TFLiteConverter.from_saved_model(saved_model_dir)
    tflite_model = converter.convert()
    
    with open(tflite_model_path, 'wb') as f:
        f.write(tflite_model)
        

def run_tf_model(tf_model_path):
    
    model = tf.saved_model.load(tf_model_path)
    model.trainable = False

    input_tensor = tf.random.uniform([1, 3, 640, 640])

    out = model(**{'input': input_tensor})
    print(out)


if __name__ == "__main__":
    onnx_path = 'testmodel.onnx'
    tf_path = 'testmodel.tf'
    tflite_path = 'testmodel.tflite'
    torch_to_onnx(onnx_path)
    onnx_to_tf(onnx_path, tf_path)
    tf_to_tflite(tf_path, tflite_path)

