import torch
import torchvision.models as models
import torch.nn as nn

from torchvision.models.video.resnet import model_urls as resnet_model_urls
resnet_model_urls['r3d_18'] = resnet_model_urls['r3d_18'].replace('https://', 'http://')

class ResNet18(nn.Module):
    def __init__(self, num_classes=2):
        super(ResNet18, self).__init__()
        self.base_net = models.resnet18(pretrained=True)
        self.num_classes = num_classes
        self.base_net.fc = nn.Linear(512, num_classes)

    def forward(self, x):
        x = self.base_net(x)
        return x, []


class ResNet3D18(nn.Module):
    def __init__(self, num_classes=2):
        super(ResNet3D18, self).__init__()
        self.base_net = models.video.r3d_18(pretrained=True, progress=True)
        self.num_classes = num_classes
        self.base_net.fc = nn.Linear(512, num_classes)

    def forward(self, x):
        x = self.base_net(x)
        return x

def load_model(model_name, num_classes=2):
    print('loading network model: %s..' % model_name)

    if model_name == 'ResNet18':
        model = ResNet18(num_classes=num_classes)
        model.output_type = 'img2type'

    elif model_name == 'ResNet3D18':
        model = ResNet3D18(num_classes=num_classes)
        model.output_type = 'video2type'

    else:
        print('ERROR: model name does not exist..')
        return

    return model


if __name__ == '__main__':

    # for images
    #img = torch.rand(1, 3, 224, 224)
    #m = ResNet18(nb_classes=5)
    vid = torch.rand(1, 3, 16, 116, 116)
    m = load_model('ResNet3D18', num_classes=2)
    out = m(vid)
    print(out)
    # optimizer = torch.optim.Adam(m.parameters(), lr=1e-4)
    # optimizer.step()
    # print(m(img))

    pass
