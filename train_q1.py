import torch
import torch.nn as nn
import torchvision.datasets as datasets
import torchvision.transforms as transforms
from sklearn.metrics import accuracy_score
from torch.functional import F
import matplotlib.pyplot as plt
import pickle


def load_data(batch_size):
    transform_train_1 = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
    ])

    transform_train_2 = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
    ])

    transform_train_3 = transforms.Compose([
        transforms.RandomRotation(10),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
    ])

    transform_train_4 = transforms.Compose([
        transforms.RandomAffine(0, shear=10, scale=(0.8, 1.2)),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
    ])

    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
    ])

    train_dataset_1 = datasets.CIFAR10(root='./data', train=True, download=True, transform=transform_train_1)
    train_dataset_2 = datasets.CIFAR10(root='./data', train=True, download=True, transform=transform_train_2)
    train_dataset_3 = datasets.CIFAR10(root='./data', train=True, download=True, transform=transform_train_3)
    train_dataset_4 = datasets.CIFAR10(root='./data', train=True, download=True, transform=transform_train_4)
    train_dataset = torch.utils.data.ConcatDataset([train_dataset_1, train_dataset_2])
    train_dataset = torch.utils.data.ConcatDataset([train_dataset, train_dataset_3])
    train_dataset = torch.utils.data.ConcatDataset([train_dataset, train_dataset_4])

    test_dataset = datasets.CIFAR10(root='./data', train=False, download=True, transform=transform_test)

    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=8,
                                               pin_memory=True)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=8,
                                              pin_memory=True)

    return train_loader, test_loader


class CIFAR10CNN(nn.Module):
    def __init__(self):
        super(CIFAR10CNN, self).__init__()

        self.conv1 = nn.Conv2d(3, 16, 3, padding=1)
        self.conv2 = nn.Conv2d(16, 32, 3, padding=1)
        self.conv3 = nn.Conv2d(32, 64, 3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)

        self.fc1 = nn.Linear(64 * 4 * 4, 25)
        self.fc2 = nn.Linear(25, 10)

        self.Dropout = nn.Dropout(0.25)
        self.loss = nn.CrossEntropyLoss()

    def forward(self, input, y):
        x = self.pool(F.relu(self.conv1(input)))
        x = self.pool(F.relu(self.conv2(x)))
        x = self.pool(F.relu(self.conv3(x)))
        x = self.Dropout(x)
        x = x.view(-1, 64 * 4 * 4)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)

        return x, self.loss(x, y)


def plot_epochs_losses_and_errors(loss_dict, error_dict):
    plt.plot(loss_dict['test'], label = "test", color = "blue")
    plt.plot(loss_dict['train'], label = "train", color = "green")
    plt.title("Loss of train and test", fontweight = "bold", fontsize = 12)
    plt.xlabel("epochs")
    plt.ylabel("loss")
    plt.legend()
    plt.show()

    plt.plot(error_dict['test'], label = "test", color = "blue")
    plt.plot(error_dict['train'], label = "train", color = "green")
    plt.plot([0.2]*len(loss_dict['train']), label = "error threshold", color ="red")
    plt.title(f"Error of train and test", fontweight = "bold", fontsize = 12)
    plt.xlabel("epochs")
    plt.ylabel("error")
    plt.legend()
    plt.show()


def learn_and_predict(model, data_loaders, optimizer, num_epochs):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    errors = {'train': [], 'test': []}
    losses = {'train': [], 'test': []}

    for epoch in range(num_epochs):
        print(f'Epoch {epoch + 1}/{num_epochs}')
        print('-' * 30)

        for phase in ['train', 'test']:
            if phase == 'train':
                model.train()
            else:
                model.eval()

            labels_list, preds_list = [], []
            running_loss = 0.0
            counter_batches = 0
            for i, (data_images, data_labels) in enumerate(data_loaders[phase]):
                counter_batches += 1
                if torch.cuda.is_available():
                    images = data_images.to(device)
                    labels = data_labels.to(device)
                else:
                    images = data_images
                    labels = data_labels

                if phase == 'train':
                    optimizer.zero_grad()
                    outputs, loss = model(images, labels)
                    loss.backward()
                    optimizer.step()

                with torch.no_grad():
                    outputs, loss = model(images, labels)

                running_loss += loss.item()
                _, predicted = torch.max(outputs.data, 1)
                labels_list += labels.cpu().view(-1).tolist()
                preds_list += predicted.view(-1).tolist()

            epoch_acc = accuracy_score(labels_list, preds_list)
            errors[phase].append(1 - epoch_acc)
            losses[phase].append(running_loss / counter_batches)

            print(f'{phase.title()}  Accuracy: {epoch_acc}')
        print()

    with open("model_q1.pkl", "wb") as f:
        pickle.dump(model, f)
    return losses, errors


def train_model_q1():
    num_epochs = 36
    batch_size = 100
    learning_rate = 0.001
    train_loader, test_loader = load_data(batch_size=batch_size)
    data_loaders = {'train': train_loader, 'test': test_loader}
    model = CIFAR10CNN()
    # print number of model parameters
    print(f'Number of model parameters: {sum(p.numel() for p in model.parameters())}')
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    losses, errors = learn_and_predict(model, data_loaders, optimizer, num_epochs)
    plot_epochs_losses_and_errors(losses, errors)
