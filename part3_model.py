# https://github.com/CuthbertCai/pytorch_DANN/blob/master/models/models.py

import torch
import torch.nn as nn
# import torch.nn.functional as F
# from torch.autograd import Variable
# import torch.nn.init as init

class GradReverse(torch.autograd.Function):
    """
    Extension of grad reverse layer
    """
    @staticmethod
    def forward(ctx, x, constant):
        ctx.constant = constant
        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad_output):
        grad_output = grad_output.neg() * ctx.constant
        return grad_output, None

    def grad_reverse(x, constant):
        return GradReverse.apply(x, constant)

# Feature Extractor
class Extractor(nn.Module):

    def __init__(self):
        super(Extractor, self).__init__()
        self.feature_extractor = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=5),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),

            nn.Conv2d(32, 48, 5),
            nn.BatchNorm2d(48),
            nn.Dropout2d(),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
        )
    def forward(self, x):
        return self.feature_extractor(x).view(-1, 48 * 4 * 4)
        
# Label predictor
class Class_classifier(nn.Module):

    def __init__(self):
        super(Class_classifier, self).__init__()
        # self.fc1 = nn.Linear(50 * 4 * 4, 100)
        # self.bn1 = nn.BatchNorm1d(100)
        # self.fc2 = nn.Linear(100, 100)
        # self.bn2 = nn.BatchNorm1d(100)
        # self.fc3 = nn.Linear(100, 10)
        # self.fc1 = nn.Linear(48 * 4 * 4, 100)
        # self.fc2 = nn.Linear(100, 100)
        # self.fc3 = nn.Linear(100, 10)
        self.classifier = nn.Sequential(
            nn.Linear(48 * 4 * 4, 100),
            nn.BatchNorm1d(100),
            nn.ReLU(inplace=True),

            nn.Linear(100, 100),
            nn.BatchNorm1d(100),
            nn.ReLU(inplace=True),

            nn.Linear(100, 10),
        )
    # def forward(self, x):
    #     # print('class classifier x:', x)
    #     # logits = F.relu(self.bn1(self.fc1(x)))
    #     # logits = self.fc2(F.dropout(logits))
    #     # logits = F.relu(self.bn2(logits))
    #     # logits = self.fc3(logits)
    #     logits = F.relu(self.fc1(x))
    #     logits = self.fc2(F.dropout(logits))
    #     logits = F.relu(logits)
    #     logits = self.fc3(logits)

    #     return F.log_softmax(logits, 1)
    def forward(self, x):
        return self.classifier(x)

# Domain Classifier
class Domain_classifier(nn.Module):

    def __init__(self):
        super(Domain_classifier, self).__init__()
        # self.fc1 = nn.Linear(50 * 4 * 4, 100)
        # self.bn1 = nn.BatchNorm1d(100)
        # self.fc2 = nn.Linear(100, 2)
        # self.fc1 = nn.Linear(48 * 4 * 4, 100)
        # self.fc2 = nn.Linear(100, 2)
        self.domain_classifier = nn.Sequential(
            nn.Linear(48 * 4 * 4, 100),
            nn.BatchNorm1d(100),
            nn.ReLU(inplace=True),
            nn.Linear(100, 2),
        )

    def forward(self, x, constant):
        x = GradReverse.grad_reverse(x, constant)
        domain_out = self.domain_classifier(x)

        # # logits = F.relu(self.bn1(self.fc1(input)))
        # # logits = F.log_softmax(self.fc2(logits), 1)
        # logits = F.relu(self.fc1(input))
        # logits = F.log_softmax(self.fc2(logits), 1)

        return domain_out


class myDANN(nn.Module):
    def __init__(self, num_classes=10):
        super(myDANN, self).__init__()
        self.fe = Extractor()
        self.cc = Class_classifier()
        self.dc = Domain_classifier()

    def forward(self, x, alpha):
        feat = self.fe(x)
        class_out = self.cc(feat)
        domain_out = self.dc(feat, alpha)
        return class_out, domain_out

    def get_feature(self, x):
        return self.fe(x)


class Non_Adaptive_model(nn.Module):
    def __init__(self, num_classes=10):
        super(Non_Adaptive_model, self).__init__()
        self.fe = Extractor()
        self.cc = Class_classifier()

    def forward(self, x):
        feat = self.fe(x)
        class_out = self.cc(feat)

        return class_out
