import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader

# 공격 기법
from torchattacks import PGD, CW, AutoAttack

# 모델 정의 (ResNet18 예제, 필요시 Vision Transformer로 변경 가능)
from torchvision.models import resnet18


def get_data_loader(batch_size=128, dataset='CIFAR10'):
    if dataset in ['MNIST', 'FashionMNIST']:
        transform = transforms.Compose([
            transforms.Grayscale(num_output_channels=3),  # 1채널 -> 3채널 변환
            transforms.Resize((32, 32)),  # ResNet에 맞게 크기 조정
            transforms.ToTensor(),
            transforms.Normalize((0.5,), (0.5,))
        ])
    else:
        transform = transforms.Compose([
            transforms.Resize((32, 32)),  # TinyImageNet 등 크기 조정
            transforms.ToTensor(),
            transforms.Normalize((0.5,), (0.5,))
        ])

    if dataset == 'CIFAR10':
        trainset = datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
        testset = datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)
    elif dataset == 'MNIST':
        trainset = datasets.MNIST(root='./data', train=True, download=True, transform=transform)
        testset = datasets.MNIST(root='./data', train=False, download=True, transform=transform)
    elif dataset == 'FashionMNIST':
        trainset = datasets.FashionMNIST(root='./data', train=True, download=True, transform=transform)
        testset = datasets.FashionMNIST(root='./data', train=False, download=True, transform=transform)
    elif dataset == 'SVHN':
        trainset = datasets.SVHN(root='./data', split='train', download=True, transform=transform)
        testset = datasets.SVHN(root='./data', split='test', download=True, transform=transform)
    elif dataset == 'TinyImageNet':
        trainset = datasets.ImageFolder(root='./data/tiny-imagenet-200/train', transform=transform)
        testset = datasets.ImageFolder(root='./data/tiny-imagenet-200/val', transform=transform)
    else:
        raise ValueError("Unsupported dataset. Implement other datasets as needed.")

    train_loader = DataLoader(trainset, batch_size=batch_size, shuffle=True, num_workers=0)
    test_loader = DataLoader(testset, batch_size=batch_size, shuffle=False, num_workers=0)

    return train_loader, test_loader



def train_adversarial(model, train_loader, optimizer, criterion, device, attack):
    model.train()
    total_loss, correct, total = 0, 0, 0

    for inputs, targets in train_loader:
        inputs, targets = inputs.to(device), targets.to(device)

        # Adversarial Attack 적용
        adv_inputs = attack(inputs, targets)

        optimizer.zero_grad()
        outputs = model(adv_inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        _, predicted = outputs.max(1)
        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()

    return total_loss / len(train_loader), 100. * correct / total


def evaluate_model(model, test_loader, criterion, device, attack=None):
    model.eval()
    total_loss, correct, total = 0, 0, 0

    for inputs, targets in test_loader:
        inputs, targets = inputs.to(device), targets.to(device)

        if attack:
            inputs.requires_grad_()  # requires_grad 활성화
            with torch.enable_grad():  # Gradient Tracking 활성화
                inputs = attack(inputs, targets)  # 공격 적용

        with torch.no_grad():  # 모델 평가 시에는 no_grad 사용
            outputs = model(inputs)
            loss = criterion(outputs, targets)

        total_loss += loss.item()
        _, predicted = outputs.max(1)
        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()

    return total_loss / len(test_loader), 100. * correct / total




def plot_results(results):
    plt.figure(figsize=(10, 5))
    plt.plot(results['epochs'], results['train_acc'], label='Train Acc')
    plt.plot(results['epochs'], results['test_acc'], label='Test Acc')
    plt.plot(results['epochs'], results['adv_acc'], label='Adversarial Acc')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy (%)')
    plt.legend()
    plt.title('Adversarial Training Performance')
    plt.show()

if __name__ == "__main__":
    # 실행 환경 설정
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 실험할 데이터셋 목록
    datasets_list = ['CIFAR10', 'MNIST', 'FashionMNIST', 'SVHN', 'TinyImageNet']
    batch_size = 128
    num_epochs = 10

    for dataset in datasets_list:
        print(f"\nTraining on {dataset} dataset")
        train_loader, test_loader = get_data_loader(batch_size, dataset)

        # 모델 정의
        model = resnet18(num_classes=10).to(device)
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(model.parameters(), lr=0.001)

        # PGD 공격 (논문에서 사용된 방식)
        pdg_attack = PGD(model, eps=8/255, alpha=2/255, steps=7)

        # 학습 및 평가
        results = {'epochs': [], 'train_acc': [], 'test_acc': [], 'adv_acc': []}

        for epoch in range(num_epochs):
            train_loss, train_acc = train_adversarial(model, train_loader, optimizer, criterion, device, pdg_attack)
            test_loss, test_acc = evaluate_model(model, test_loader, criterion, device)
            adv_loss, adv_acc = evaluate_model(model, test_loader, criterion, device, attack=pdg_attack)

            results['epochs'].append(epoch + 1)
            results['train_acc'].append(train_acc)
            results['test_acc'].append(test_acc)
            results['adv_acc'].append(adv_acc)

            print(f"Epoch {epoch+1}/{num_epochs}: Train Acc={train_acc:.2f}%, Test Acc={test_acc:.2f}%, Adv Acc={adv_acc:.2f}%")

        # 결과 시각화
        plot_results(results)

        # AutoAttack, CW Attack 등 추가 적용 가능
        cw_attack = CW(model, c=1, kappa=0, steps=100)
        auto_attack = AutoAttack(model, norm='Linf', eps=8/255, version='standard')

        cw_loss, cw_acc = evaluate_model(model, test_loader, criterion, device, attack=cw_attack)
        auto_loss, auto_acc = evaluate_model(model, test_loader, criterion, device, attack=auto_attack)

        print(f"CW Attack Acc on {dataset}: {cw_acc:.2f}%")
        print(f"AutoAttack Acc on {dataset}: {auto_acc:.2f}%")
