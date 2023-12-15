import torch
import torchvision.datasets as datasets
import torchvision.transforms as transforms
from sklearn.metrics import accuracy_score
import pickle
import io
from train_q1 import CIFAR10CNN


def load_data(batch_size):
    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
    ])

    test_dataset = datasets.CIFAR10(root='./data', train=False, download=True, transform=transform_test)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=8, pin_memory=True)

    return test_loader


class CPU_Unpickler(pickle.Unpickler):
    def find_class(self, module, name):
        if module == 'torch.storage' and name == '_load_from_bytes':
            return lambda b: torch.load(io.BytesIO(b), map_location=lambda storage, loc: storage)
        else:
            return super().find_class(module, name)


def evaluate_model_q1():
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    if device == "cuda:0":
        model = pickle.load(open('model_q1.pkl', 'rb'))
    else:
        model = CPU_Unpickler(open('model_q1.pkl', 'rb')).load()

    model.to(device)
    model.eval()
    test_loader = load_data(batch_size=100)
    preds_list, labels_list = [], []

    for i, (data_images, data_labels) in enumerate(test_loader):
        if torch.cuda.is_available():
            images = data_images.to(device)
            labels = data_labels.to(device)
        else:
            images = data_images
            labels = data_labels

        with torch.no_grad():
            outputs, _ = model(images, labels)

        _, predicted = torch.max(outputs.data, 1)
        preds_list += predicted.view(-1).tolist()
        labels_list += labels.cpu().view(-1).tolist()

    return round(1-accuracy_score(labels_list, preds_list), 4)
