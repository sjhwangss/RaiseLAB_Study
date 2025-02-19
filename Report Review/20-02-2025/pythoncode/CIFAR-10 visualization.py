import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
import numpy as np

# 공격 기법 (PGD만 사용)
from torchattacks import PGD

# 모델 정의 (ResNet18 예제)
from torchvision.models import resnet18


def get_data_loader(batch_size=128, transform_type='augment'):
    """
    CIFAR-10 데이터셋에 대해 여러 전처리 방식을 선택할 수 있습니다.

    transform_type 옵션:
      - 'baseline': ToTensor와 Normalize만 적용
      - 'augment': RandomCrop, RandomHorizontalFlip 적용
      - 'autoaugment': AutoAugment 정책(CIFAR10) 적용 (Torchvision 0.9 이상 필요)
      - 'randaugment': RandAugment 적용 (Torchvision 0.9 이상 필요)
      - 'colorjitter': ColorJitter로 색상 변형 적용
    """
    # CIFAR-10의 평균과 표준편차
    mean = (0.4914, 0.4822, 0.4465)
    std = (0.2023, 0.1994, 0.2010)

    if transform_type == 'baseline':
        transform_train = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean, std)
        ])
    elif transform_type == 'augment':
        transform_train = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(mean, std)
        ])
    elif transform_type == 'autoaugment':
        transform_train = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.AutoAugment(policy=transforms.AutoAugmentPolicy.CIFAR10),
            transforms.ToTensor(),
            transforms.Normalize(mean, std)
        ])
    elif transform_type == 'randaugment':
        transform_train = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.RandAugment(),
            transforms.ToTensor(),
            transforms.Normalize(mean, std)
        ])
    elif transform_type == 'colorjitter':
        transform_train = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
            transforms.ToTensor(),
            transforms.Normalize(mean, std)
        ])
    else:
        raise ValueError(
            "Unknown transform type. Please choose among 'baseline', 'augment', 'autoaugment', 'randaugment', or 'colorjitter'.")

    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean, std)
    ])

    trainset = datasets.CIFAR10(root='./data', train=True, download=True, transform=transform_train)
    testset = datasets.CIFAR10(root='./data', train=False, download=True, transform=transform_test)

    train_loader = DataLoader(trainset, batch_size=batch_size, shuffle=True, num_workers=0)
    test_loader = DataLoader(testset, batch_size=batch_size, shuffle=False, num_workers=0)

    return train_loader, test_loader


def train_adversarial(model, train_loader, optimizer, criterion, device, attack):
    model.train()
    total_loss, correct, total = 0, 0, 0

    for inputs, targets in train_loader:
        inputs, targets = inputs.to(device), targets.to(device)

        # PGD 공격 적용
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
            inputs.requires_grad_()  # Gradient 추적 활성화
            with torch.enable_grad():
                inputs = attack(inputs, targets)  # 공격 적용

        with torch.no_grad():
            outputs = model(inputs)
            loss = criterion(outputs, targets)

        total_loss += loss.item()
        _, predicted = outputs.max(1)
        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()

    return total_loss / len(test_loader), 100. * correct / total


def plot_aggregated_results(results_list, transform_type):
    """
    여러 번 실행한 결과(리스트)를 겹쳐서 플롯합니다.
    results_list: 각 실행 결과를 담은 리스트 (각 결과는 dict 형태)
    """
    epochs = results_list[0]['epochs']  # 모든 실행의 에포크는 동일하다고 가정
    num_runs = len(results_list)

    # 각 실행별 결과 배열로 변환
    train_acc_all = np.array([r['train_acc'] for r in results_list])
    test_acc_all  = np.array([r['test_acc'] for r in results_list])
    adv_acc_all   = np.array([r['adv_acc'] for r in results_list])

    # 평균 계산
    train_acc_mean = np.mean(train_acc_all, axis=0)
    test_acc_mean  = np.mean(test_acc_all, axis=0)
    adv_acc_mean   = np.mean(adv_acc_all, axis=0)

    # 3개의 서브플롯: Train, Test, Adversarial Accuracy
    fig, axs = plt.subplots(1, 3, figsize=(18, 5))
    for i in range(num_runs):
        axs[0].plot(epochs, train_acc_all[i], color='blue', alpha=0.3)
        axs[1].plot(epochs, test_acc_all[i], color='green', alpha=0.3)
        axs[2].plot(epochs, adv_acc_all[i], color='red', alpha=0.3)
    axs[0].plot(epochs, train_acc_mean, color='blue', linewidth=2, label='Average')
    axs[1].plot(epochs, test_acc_mean, color='green', linewidth=2, label='Average')
    axs[2].plot(epochs, adv_acc_mean, color='red', linewidth=2, label='Average')

    axs[0].set_title('Train Accuracy')
    axs[1].set_title('Test Accuracy')
    axs[2].set_title('Adversarial Accuracy')
    for ax in axs:
        ax.set_xlabel('Epoch')
        ax.set_ylabel('Accuracy (%)')
        ax.legend()
    plt.suptitle(f'Aggregated Performance for transform: {transform_type}', fontsize=16)
    plt.show()


def plot_runs_subplots(results_list, transform_type):
    """
    각 전처리 방식에 대해 10번의 실행 결과를 하나의 서브플롯으로 출력합니다.
    results_list: 각 실행 결과를 담은 리스트 (각 결과는 dict 형태)
    """
    num_runs = len(results_list)
    cols = 5  # 10번이면 5열 x 2행
    rows = (num_runs + cols - 1) // cols

    fig, axes = plt.subplots(rows, cols, figsize=(cols * 4, rows * 3))
    axes = axes.flatten()

    for i, result in enumerate(results_list):
        ax = axes[i]
        epochs = result['epochs']
        ax.plot(epochs, result['train_acc'], label='Train', marker='o')
        ax.plot(epochs, result['test_acc'], label='Test', marker='o')
        ax.plot(epochs, result['adv_acc'], label='Adv', marker='o')
        ax.set_title(f"Run {i+1}")
        ax.set_xlabel("Epoch")
        ax.set_ylabel("Accuracy (%)")
        ax.legend(fontsize=8)
    # 남는 서브플롯 없으면 제거
    for j in range(i+1, len(axes)):
        fig.delaxes(axes[j])
    fig.suptitle(f"{transform_type} - 10 Runs", fontsize=16)
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.show()


if __name__ == "__main__":
    # 실행 환경 설정 (GPU 사용 가능시 GPU 사용)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    batch_size = 128
    num_epochs = 10
    # 사용 가능한 transform_type: 'baseline', 'augment', 'autoaugment', 'randaugment', 'colorjitter'
    transform_types = ['baseline', 'augment', 'autoaugment', 'randaugment', 'colorjitter']
    num_repeats = 10  # 각 전처리 방식당 반복 횟수

    # 전체 결과를 저장할 딕셔너리 (나중에 분석에도 활용 가능)
    all_results = {}

    for transform_type in transform_types:
        print("\n==============================================")
        print(f"Training with transform type: {transform_type}")
        print("==============================================")

        results_runs = []  # 해당 전처리 방식에 대해 반복 실행한 결과 저장

        for run in range(num_repeats):
            print(f"Run {run+1}/{num_repeats} for {transform_type}")
            train_loader, test_loader = get_data_loader(batch_size=batch_size, transform_type=transform_type)

            # 모델, 손실 함수, 옵티마이저 정의
            model = resnet18(num_classes=10).to(device)
            criterion = nn.CrossEntropyLoss()
            optimizer = optim.Adam(model.parameters(), lr=0.001)

            # PGD 공격 설정
            pgd_attack = PGD(model, eps=8 / 255, alpha=2 / 255, steps=7)

            results = {'epochs': [], 'train_acc': [], 'test_acc': [], 'adv_acc': []}

            for epoch in range(num_epochs):
                train_loss, train_acc = train_adversarial(model, train_loader, optimizer, criterion, device, pgd_attack)
                test_loss, test_acc = evaluate_model(model, test_loader, criterion, device)
                adv_loss, adv_acc = evaluate_model(model, test_loader, criterion, device, attack=pgd_attack)

                results['epochs'].append(epoch + 1)
                results['train_acc'].append(train_acc)
                results['test_acc'].append(test_acc)
                results['adv_acc'].append(adv_acc)

                print(f"[{transform_type}][Run {run+1}] Epoch {epoch+1}/{num_epochs}: Train Acc = {train_acc:.2f}%, Test Acc = {test_acc:.2f}%, Adv Acc = {adv_acc:.2f}%")

            results_runs.append(results)

        all_results[transform_type] = results_runs

        # 10번 실행 결과를 하나의 서브플롯으로 출력 (각 실행별 개별 서브플롯)
        plot_runs_subplots(results_runs, transform_type)

        # 해당 전처리 방식에 대한 10회 실행 결과를 겹쳐서 하나의 aggregated 그래프로 출력
        plot_aggregated_results(results_runs, transform_type)

