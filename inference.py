import argparse
import torch
import os
import torchvision.transforms as transforms
from tqdm import tqdm
import shutil
from mobilenetv3 import MobileNetV3_Small, MobileNetV3_Large
from torch.utils.data import Dataset
import utils
from PIL import Image
import torch.nn.functional as F
from pathlib import Path

def get_args_parser():
    parser = argparse.ArgumentParser('ConvNeXt training and evaluation script for image classification', add_help=False)
    parser.add_argument('--data_path', default='/home/cw/Desktop/liu/dataset/MobileNetv3/val/bg', type=str,
                        help='inference data root')
    parser.add_argument('--ckpt', default='ckpt0523/checkpoint-best.pth', type=str, help='ckpt for model inference')
    parser.add_argument('--output_dir', default='0523_test', type=str,
                        help='dictionary to save inference result ')
    parser.add_argument('--device', default='cuda',
                    help='device to use for inference')
    parser.add_argument('--batch_size', default=16, type=int,
                    help='Per GPU batch size')
    
    return parser

class infDataset(Dataset):
    def __init__(self, root_dir: str) -> None:
        super(infDataset, self).__init__()
        self.root_dir = root_dir
        self.images = os.listdir(root_dir)
        transforms_ = [
            transforms.Resize((800, 800)),
            transforms.ToTensor(),
            transforms.Normalize((0.5,), (0.5,))  # (x-mean) / std
        ]
        self.transforms = transforms.Compose(transforms_)
    
    def __len__(self):
        return len(self.images)
    
    def __getitem__(self, index) :
        img_path = os.path.join(self.root_dir, self.images[index])
        img = Image.open(img_path).convert("L")  # 需要识别的图像读为灰度
        img = self.transforms(img)

        return (img_path, img)

def main(args):
    print(args)
    device = torch.device(args.device)

    dataset = infDataset(args.data_path)
    dataloder = torch.utils.data.DataLoader(
        dataset, shuffle=True,
        batch_size=args.batch_size,
        num_workers=4,
        drop_last=True,
    )

    model = MobileNetV3_Small(num_classes=2)
    if args.ckpt:
        checkpoint = torch.load(args.ckpt, map_location='cpu')
        utils.load_state_dict(model, checkpoint['model'])
    else:
        print("there is no ckpt!!!")
        return
    
    model.to(device)
    model.eval()

    for img_path, img in tqdm(dataloder):
        
        img = img.to(device, non_blocking=True)
        
        output = model(img)
        
        softmax_output = F.softmax(output, dim=1)
        cls_output = torch.argmax(softmax_output, dim=1)

        for i in range(args.batch_size):
            img_path_batch = img_path[i]
            img_name = img_path_batch.split('/')[-1].split('.')[0]
            shutil.copy(img_path[i], os.path.join(args.output_dir, f'{img_name}_{cls_output[i].item()}_{softmax_output[i][cls_output[i]].item():.2f}.jpg'))

    print("------Inferenc Ending-----")

if __name__ == '__main__':
    parser = argparse.ArgumentParser('ConvNeXt training and evaluation script', parents=[get_args_parser()])
    args = parser.parse_args()
    if args.output_dir:
        Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    main(args)