import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import numpy as np
from torchvision.models import resnet18, alexnet

# Device 설정
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# CIFAR-10 클래스 이름 정의
class_names = [
    "airplane", "automobile", "bird", "cat", "deer",
    "dog", "frog", "horse", "ship", "truck"
]

# 데이터 변환 (각 모델에 맞게)
transform_224 = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

transform_32 = transforms.Compose([
    transforms.Resize((32, 32)),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

# CIFAR-10 데이터셋 로드
test_dataset_224 = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform_224)
test_loader_224 = torch.utils.data.DataLoader(test_dataset_224, batch_size=64, shuffle=False)

test_dataset_32 = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform_32)
test_loader_32 = torch.utils.data.DataLoader(test_dataset_32, batch_size=64, shuffle=False)

# 간단한 CNN 모델 정의
class SimpleCNN(nn.Module):
    def __init__(self):
        super(SimpleCNN, self).__init__()
        self.conv_layers = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Dropout(0.25)
        )
        self.fc_layers = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64 * 16 * 16, 128),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(128, 10)
        )

    def forward(self, x):
        x = self.conv_layers(x)
        x = self.fc_layers(x)
        return x

# FGSM 공격 함수 (부분 공격 포함)
def fgsm_attack(image, epsilon, gradient, region="full"):
    h, w = image.size(2), image.size(3)
    h_half, w_half = h // 2, w // 2
    perturbed_image = image.clone()

    if region == "top_left":
        perturbed_image[:, :, :h_half, :w_half] += epsilon * gradient.sign()[:, :, :h_half, :w_half]
    elif region == "top_right":
        perturbed_image[:, :, :h_half, w_half:] += epsilon * gradient.sign()[:, :, :h_half, w_half:]
    elif region == "bottom_left":
        perturbed_image[:, :, h_half:, :w_half] += epsilon * gradient.sign()[:, :, h_half:, :w_half]
    elif region == "bottom_right":
        perturbed_image[:, :, h_half:, w_half:] += epsilon * gradient.sign()[:, :, h_half:, w_half:]
    elif region == "center":
        perturbed_image[:, :, h_half // 2:-h_half // 2, w_half // 2:-w_half // 2] += \
            epsilon * gradient.sign()[:, :, h_half // 2:-h_half // 2, w_half // 2:-w_half // 2]
    elif region == "border":
        perturbed_image[:, :, :h_half // 2, :] += epsilon * gradient.sign()[:, :, :h_half // 2, :]
        perturbed_image[:, :, -h_half // 2:, :] += epsilon * gradient.sign()[:, :, -h_half // 2:, :]
        perturbed_image[:, :, :, :w_half // 2] += epsilon * gradient.sign()[:, :, :, :w_half // 2]
        perturbed_image[:, :, :, -w_half // 2:] += epsilon * gradient.sign()[:, :, :, -w_half // 2:]
    else:  # Full attack
        perturbed_image += epsilon * gradient.sign()

    perturbed_image = torch.clamp(perturbed_image, -1, 1)
    return perturbed_image

# 모델 로드 (weights_only=True 적용)
resnet_model = resnet18(weights=None)
resnet_model.fc = nn.Linear(resnet_model.fc.in_features, 10)
resnet_model.load_state_dict(torch.load('resnet18_cifar10.pth', weights_only=True))
resnet_model = resnet_model.to(device)
resnet_model.eval()

alexnet_model = alexnet(weights=None)
alexnet_model.classifier[6] = nn.Linear(4096, 10)
alexnet_model.load_state_dict(torch.load('alexnet_cifar10.pth', weights_only=True))
alexnet_model = alexnet_model.to(device)
alexnet_model.eval()

cnn_model = SimpleCNN().to(device)
cnn_model.load_state_dict(torch.load('cnn_cifar10.pth', weights_only=True))
cnn_model.eval()

# 손실 함수
criterion = nn.CrossEntropyLoss()

# 적대적 샘플 생성 후 모델 평가
def evaluate_adversarial_accuracy(model, test_loader, epsilon, region, device):
    model.eval()
    correct = 0
    total = 0

    for images, labels in test_loader:
        images, labels = images.to(device), labels.to(device)

        # ✅ FGSM 공격을 위해 requires_grad=True 설정
        images.requires_grad = True

        # 원본 예측
        outputs = model(images)
        loss = criterion(outputs, labels)

        # ✅ 그래디언트 초기화 후 역전파 수행
        model.zero_grad()
        loss.backward()

        # ✅ FGSM 공격 수행
        gradient = images.grad
        adversarial_images = fgsm_attack(images, epsilon, gradient, region)

        # 적대적 샘플 평가
        with torch.no_grad():  # 평가 시에는 그래디언트 불필요
            adv_outputs = model(adversarial_images)
            _, predicted = torch.max(adv_outputs, 1)

            correct += (predicted == labels).sum().item()
            total += labels.size(0)

    accuracy = 100 * correct / total
    return accuracy

# FGSM 공격 강도 (Epsilon 값)
epsilons = [0, 0.001, 0.002, 0.003, 0.004, 0.005, 0.01, 0.02, 0.03, 0.04, 0.05, 0.1, 0.2, 0.5, 1.0]
attack_regions = ["full", "top_left", "top_right", "bottom_left", "bottom_right", "center", "border"]

# 정확도 저장을 위한 딕셔너리 초기화
cnn_acc_results = {region: [] for region in attack_regions}
alexnet_acc_results = {region: [] for region in attack_regions}
resnet_acc_results = {region: [] for region in attack_regions}

# FGSM 공격 후 성능 평가 (모든 부분 공격 포함)
for epsilon in epsilons:
    print(f"=== FGSM Attack (ε={epsilon}) ===")

    for region in attack_regions:
        cnn_acc = evaluate_adversarial_accuracy(cnn_model, test_loader_32, epsilon, region, device)
        alexnet_acc = evaluate_adversarial_accuracy(alexnet_model, test_loader_224, epsilon, region, device)
        resnet_acc = evaluate_adversarial_accuracy(resnet_model, test_loader_224, epsilon, region, device)

        cnn_acc_results[region].append(cnn_acc)
        alexnet_acc_results[region].append(alexnet_acc)
        resnet_acc_results[region].append(resnet_acc)

        # 실행창 출력
        print(f"   [{region.upper()}] CNN 정확도: {cnn_acc:.2f}%")
        print(f"   [{region.upper()}] AlexNet 정확도: {alexnet_acc:.2f}%")
        print(f"   [{region.upper()}] ResNet-18 정확도: {resnet_acc:.2f}%")
        print("-" * 40)

# 그래프 생성 및 저장
for region in attack_regions:
    plt.figure(figsize=(8, 6))
    plt.plot(epsilons, cnn_acc_results[region], marker='o', label="CNN")
    plt.plot(epsilons, alexnet_acc_results[region], marker='s', label="AlexNet")
    plt.plot(epsilons, resnet_acc_results[region], marker='^', label="ResNet-18")
    plt.xlabel("FGSM Epsilon (ε)")
    plt.ylabel("Accuracy (%)")
    plt.title(f"Model Accuracy under FGSM Attack ({region.upper()})")
    plt.legend()
    plt.grid(True)
    plt.ylim(0, 100)
    plt.savefig(f"fgsm_accuracy_{region}.png")
    plt.show()

