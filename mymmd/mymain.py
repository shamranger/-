import sys
import os
import argparse
import timeit

import torch
import torchvision.transforms as transforms
from torchvision.models import vgg19
from torch.utils.data import Dataset
import torch.nn as nn
from torch.optim import Adam

from mydataset import SingleStyleData
from mymodel import VGG
from mymmlosses import ContentLoss,StyleLoss

vgg = vgg19(pretrained=False).eval()
#print(type(vgg))

net = torch.load('./weights/vgg19-dcbb9e9d.pth')
#print(type(net))
#print(len(net))
vgg.load_state_dict(net,strict=False)
#print(list(vgg.children()))
modules = list(vgg.children())[0][:29]
#print(list(vgg.children())[1])
#print(modules)
for i, layer in enumerate(modules):
    if isinstance(layer, nn.ReLU):
        modules[i] = nn.ReLU(inplace=False)
    #print(i,layer)
#print(modules)
vgg_style = VGG(modules).cuda()
for p in vgg_style.parameters():
    p.requires_grad = False
    #print(p)

parser = argparse.ArgumentParser()
parser.add_argument('--epsilon', type=float, default=0.001, help='Delta difference stopping criterion')
parser.add_argument('--max_iter', type=int, default=1000, help='Maximum iteration if delta not reached')
parser.add_argument('--alpha', type=float, default=1.0, help='Style and content loss weighting.')
parser.add_argument('--lr', type=float, default=0.1, help='Learning rate used during optimization'
                                                          'Needs to be high since we are starting with content'
                                                          'image.')
parser.add_argument('--c_img', type=str, default='./images/content/pablo_picasso.jpg', help='Path to content image')
parser.add_argument('--s_img', type=str, default='./images/style/picasso.jpg', help='Path to style image')
parser.add_argument('--im_size', type=int, default=360, nargs='+', help='Image size. Either single int or tuple of int')
args = parser.parse_args()
vgg_weights = [1e3/n**2 for n in [64,128,256,512,512]]
#print(vgg_weights)
img_size = args.im_size
#print(img_size)
if isinstance(img_size, list):
    if len(args.im_size)>2:
        print("Image size can either be a single int or a list of two ints.")
        sys.exit(0)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
p#rint('device=',device)
mean = [0.485, 0.456, 0.406]
std = [0.229, 0.224, 0.225]
content_img = args.c_img
#print('content_img=',content_img)
style_img = args.s_img
#print('style_img=',style_img)
transform = transforms.Compose([transforms.Resize(img_size),
                                transforms.ToTensor(),
                                transforms.Normalize(mean, std)])
#print('transform=',transform)
dataset = SingleStyleData(path_content=content_img,
                          path_style=style_img,
                          device=device,
                          transform=transform)
#print('dataset=',dataset)
input_img = dataset[0][0].clone().to(device)
#print(dataset[1])
#print('input_img=',input_img)
#print(input_img.size())
#print(len(dataset))
model = vgg_style
#print('model=',model)
lr = args.lr
#print('lr=',lr)
optim = Adam([input_img.requires_grad_()], lr=lr)
#print('optim=',optim)


class TensorDataset(Dataset):
    # TensorDataset继承Dataset, 重载了__init__, __getitem__, __len__
    # 实现将一组Tensor数据对封装成Tensor数据集
    # 能够通过index得到数据集的数据，能够通过len，得到数据集大小

    def __init__(self, data_tensor, target_tensor):
        self.data_tensor = data_tensor
        self.target_tensor = target_tensor

    def __getitem__(self, index):
        return self.data_tensor[index], self.target_tensor[index]

    def __len__(self):
        return self.data_tensor.size(0)    # size(0) 返回当前张量维数的第一维

# 生成数据
data_tensor = torch.randn(4, 3)   # 4 行 3 列，服从正态分布的张量
#print(data_tensor)
target_tensor = torch.rand(4)     # 4 个元素，服从均匀分布的张量
#print(target_tensor)

# 将数据封装成 Dataset （用 TensorDataset 类）
tensor_dataset = TensorDataset(data_tensor, target_tensor)

# 可使用索引调用数据
#print('tensor_data[0]: ', tensor_dataset[1])

# 可返回数据len
#print('len os tensor_dataset: ', len(tensor_dataset))

img_c, img_s = model(dataset[0][0]), model(dataset[0][1])
#print(img_c, img_s)
conv_c = torch.nn.ReLU()(img_c[3]).detach()
#print(conv_c)
convs_s = [torch.sigmoid(s).detach() for s in img_s]
#print(convs_s)


style_loss_weights=(1, 1, 1, 1, 1)
epsilon=0.0001
max_iter=1000
alpha=0.9


c_loss = ContentLoss().cuda()
s_losses = [StyleLoss(conv_s, k=len(style_loss_weights), weights=style_loss_weights).cuda() for conv_s in convs_s]
#print(type(c_loss))
#print(s_losses)
moving_loss = None
iteration = 0
starttime = timeit.default_timer()

while iteration < max_iter:
    # Feed current image into model
    outputs = model(input_img)
    loss_c = c_loss(torch.nn.ReLU()(outputs[3]), conv_c)
    loss_s = torch.tensor(0.0).cuda()
    for i, (conv_o, conv_s) in enumerate(zip(outputs, convs_s)):
        loss_s += vgg_weights[i] * s_losses[i](torch.sigmoid(conv_o))
    loss = (1 - alpha) * loss_c + alpha * loss_s
    # Optimize
    optim.zero_grad()
    loss.backward()
    optim.step()
    new_loss = loss_s.item()
    iteration += 1
    if moving_loss is None:
        moving_loss = loss_s.item()
        continue
    else:
        moving_loss = 0.99 * moving_loss + 0.01 * new_loss
    if iteration % 50 == 0:
        print("Current iteration is {} and the loss is {}".format(iteration, moving_loss))
    if abs(moving_loss - new_loss) <= epsilon:
        print("Delta smaller than eps.")
        print("Current iteration is {} and the loss is {}".format(iteration, moving_loss))
        break
difference = timeit.default_timer() - starttime
print("The time difference is :", difference)

image = input_img.clone().detach().cpu()  # we clone the tensor to not do changes on it
image = image.squeeze(0)  # remove the fake batch dimension
image = image * torch.tensor(std).view(-1, 1, 1) + torch.tensor(mean).view(-1, 1, 1)
image.data.clamp_(0, 1)
stylized_image = transforms.ToPILImage()(image)
if False:
    fig = plt.figure(figsize=(10, 10), facecolor='white')
    ax = plt.Axes(fig, [0., 0., 1., 1.])
    plt.axis('off')
    plt.title('Stylized image')
    plt.imshow(stylized_image)
    plt.show()
else:
    saveName = os.path.splitext(os.path.basename(args.c_img))[0] + '-' + os.path.splitext(os.path.basename(args.s_img))[
        0] + '_alpha='+str(alpha)+".png"
    stylized_image.save(saveName, "PNG")
