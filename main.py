from multiprocessing.spawn import freeze_support

from models.ClassifierLenet import LeNetClassifier
from models.ExtractorLenet import LeNetEncoder
from models.ExtractorVgg import vgg_Extractor
from models.ClassifierVgg import ClassifierVgg
from train_test import train, test
from utils import *
import torchvision
import os
import torchvision.models.alexnet


transform = torchvision.transforms.Compose([
    torchvision.transforms.Resize(28),
    torchvision.transforms.ToTensor()
])

src_train_data = get_data("mnist", transform=transform)
tgt_train_data = get_data("usps", transform=transform)
src_test_data = get_data("mnist", transform=transform, is_train=False)
tgt_test_data = get_data("usps", transform=transform, is_train=False)
extractor = LeNetEncoder()
classifier1 = LeNetClassifier()
classifier2 = LeNetClassifier()

extractor.apply(init_weights)
classifier1.apply(init_weights)
classifier2.apply(init_weights)

if __name__ == '__main__':
    freeze_support()
    for i in range(params.num_epoch):
        print("epoch={}".format(i))
        extractor, classifier1, classifier2 = train(src_train_data, tgt_train_data, extractor, classifier1, classifier2)

        torch.save(extractor, os.path.join(params.models_save, "extractor.pth"))
        torch.save(classifier1, os.path.join(params.models_save, "classifier1.pth"))
        torch.save(classifier2, os.path.join(params.models_save, "classifier2.pth"))

        # 源域测试
        test(src_test_data, extractor, classifier1, classifier2)
        # 目标域测试
        test(tgt_test_data, extractor, classifier1, classifier2)




