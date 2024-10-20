import torch
from torch import nn
from torchvision.models import resnet50, densenet121
import gdown

class CombinedModel(nn.Module):
    def __init__(self, num_classes):
        super(CombinedModel, self).__init__()
        self.resnet = resnet50(pretrained=False)
        self.densenet = densenet121(pretrained=False)
        self.resnet.fc = nn.Linear(self.resnet.fc.in_features, 512)
        self.densenet.classifier = nn.Linear(self.densenet.classifier.in_features, 512)
        self.fc1 = nn.Linear(1024, 512)
        self.fc2 = nn.Linear(512, num_classes)

    def forward(self, x):
        resnet_features = self.resnet(x)
        densenet_features = self.densenet(x)
        combined_features = torch.cat((resnet_features, densenet_features), dim=1)
        x = self.fc1(combined_features)
        x = nn.ReLU()(x)
        x = self.fc2(x)
        return x

def load_combined_model(num_classes):
    gdown.download('https://drive.google.com/uc?id=1pmZf86M8ixCAvNXanKnNTjumQGSjC8NO', 'combined5.pth', quiet=False)
    model = CombinedModel(num_classes=num_classes)
    model.load_state_dict(torch.load('combined5.pth', map_location=torch.device('cpu')))
    model.eval()
    return model

# Example usage
model = load_combined_model(num_classes=23)  # Adjust num_classes as needed
print(model)
