import torch
import torchvision
from torch.autograd import Variable
import netron
from mobilenetv3 import MobileNetV3_Small, MobileNetV3_Large
# # torch  -->  onnx
# input_name = ['input']
# output_name = ['output']
# input = Variable(torch.randn(1, 1, 800, 800)).cuda()
# ckpt = torch.load("ckpt/checkpoint-best.pth", map_location="cuda:0")
# model = MobileNetV3_Small(num_classes=2)
# model.load_state_dict(ckpt)
# torch.onnx.export(model, input, 'model_onnx.onnx',opset_version=12, input_names=input_name, output_names=output_name, verbose=True)


torch.set_grad_enabled(False)
torch_model = MobileNetV3_Small(num_classes=2)  # 初始化网络
torch_model.load_state_dict(torch.load('ckpt0523/checkpoint-best.pth', map_location='cpu')['model'], True)  # 加载训练好的pth模型
batch_size = 1  # 批处理大小
input_shape = (1, 800, 800)  # 输入数据,我这里是灰度训练所以1代表是单通道，RGB训练是3，128是图像输入网络的尺寸

# set the model to inference mode
torch_model.eval().cpu()  # cpu推理

x = torch.randn(batch_size, *input_shape).cpu()  # 生成张量
export_onnx_file = "ckpt0523/checkpoint-best.onnx"  # 要生成的ONNX文件名
# torch.onnx.export(torch_model,
#                   x,
#                   export_onnx_file,
#                   opset_version=10,
#                   do_constant_folding=True,  # 是否执行常量折叠优化
#                   input_names=["input"],  # 输入名
#                   output_names=["output"],  # 输出名
#                   dynamic_axes={"input": {0: "batch_size"},  # 批处理变量
#                                 "output": {0: "batch_size"}})
torch.onnx.export(torch_model,
                  x,
                  export_onnx_file,
                  opset_version=10,
                  do_constant_folding=True,  # 是否执行常量折叠优化
                  input_names=["input"],  # 输入名
                  output_names=["output"],  # 输出名
                 )

# 模型可视化
# netron.start('./ckpt0511/mobilenetv3.onnx')