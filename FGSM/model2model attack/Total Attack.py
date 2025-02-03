import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import numpy as np
from torchvision.models import resnet18, alexnet

# Device 설정
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)

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
test_loader_224 = torch.utils.data.DataLoader(test_dataset_224, batch_size=1, shuffle=False)

test_dataset_32 = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform_32)
test_loader_32 = torch.utils.data.DataLoader(test_dataset_32, batch_size=1, shuffle=False)

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

# FGSM 부분 공격 함수
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

# 모델 로드
resnet_model = resnet18(pretrained=True)
resnet_model.fc = nn.Linear(resnet_model.fc.in_features, 10)
resnet_model.load_state_dict(torch.load('resnet18_cifar10.pth'))
resnet_model = resnet_model.to(device)
resnet_model.eval()

alexnet_model = alexnet(pretrained=True)
alexnet_model.classifier[6] = nn.Linear(4096, 10)
alexnet_model.load_state_dict(torch.load('alexnet_cifar10.pth'))
alexnet_model = alexnet_model.to(device)
alexnet_model.eval()

cnn_model = SimpleCNN().to(device)
cnn_model.load_state_dict(torch.load('cnn_cifar10.pth'))
cnn_model.eval()

# 손실 함수
criterion = nn.CrossEntropyLoss()

# 적대적 샘플 생성 및 테스트
def generate_and_visualize(model, target_models, test_loader, epsilons, regions, class_names):
    model.eval()
    images, labels = next(iter(test_loader))  # 첫 번째 배치 가져오기
    images, labels = images.to(device), labels.to(device)

    for epsilon in epsilons:
        for region in regions:
            images.requires_grad = True

            # FGSM 공격 수행
            outputs = model(images)
            loss = criterion(outputs, labels)
            model.zero_grad()
            loss.backward()
            gradient = images.grad.data

            adversarial_images = fgsm_attack(images, epsilon, gradient, region)
            images.requires_grad = False

            results = {}
            for target_name, target_model in target_models.items():
                # CNN이 적대적 샘플을 만들었을 경우, 다른 모델(AlexNet/ResNet) 입력 크기에 맞게 변환
                if isinstance(model, SimpleCNN):
                    resized_images = torch.nn.functional.interpolate(adversarial_images, size=(224, 224), mode='bilinear', align_corners=False)
                # AlexNet 또는 ResNet이 적대적 샘플을 만들었을 경우, CNN 입력 크기에 맞게 변환
                elif target_name == "CNN":
                    resized_images = torch.nn.functional.interpolate(adversarial_images, size=(32, 32), mode='bilinear', align_corners=False)
                else:
                    resized_images = adversarial_images  # AlexNet & ResNet-18 간 변환 불필요

                outputs = target_model(resized_images)
                _, predicted = torch.max(outputs, 1)
                results[target_name] = class_names[predicted.item()]

            visualize_images(images[0].cpu(), adversarial_images[0].cpu(), labels[0].cpu(), epsilon, region, results, class_names)




# 시각화 함수 (양옆 배치 + 예측 결과 포함)
def visualize_images(original, adversarial, label, epsilon, region, results, class_names):
    plt.figure(figsize=(12, 4))
    plt.subplot(1, 2, 1)
    plt.imshow(np.transpose((original.detach().numpy() * 0.5 + 0.5), (1, 2, 0)))
    plt.title(f"Original: {class_names[label]}")
    plt.axis('off')

    plt.subplot(1, 2, 2)
    plt.imshow(np.transpose((adversarial.detach().numpy() * 0.5 + 0.5), (1, 2, 0)))
    plt.title(f"Adv (Eps: {epsilon}, {region})\nCNN: {results.get('CNN', 'N/A')}\nAlexNet: {results.get('AlexNet', 'N/A')}\nResNet-18: {results.get('ResNet-18', 'N/A')}")
    plt.axis('off')

    plt.tight_layout()
    save_path = f"result_files/adv_eps{epsilon}_{region}.png"
    plt.savefig(save_path)  # 이미지 저장
    plt.show()


# 실행할 FGSM 공격 강도 및 부분 공격 영역
epsilons = [0, 0.001, 0.002, 0.003, 0.004, 0.005,0.01, 0.02, 0.03, 0.04, 0.05, 0.1, 0.2, 0.5, 1.0]
attack_regions = ["full", "top_left", "top_right", "bottom_left", "bottom_right", "center", "border"]

# CNN이 공격 수행 → AlexNet, ResNet-18 테스트
print("=== CNN → AlexNet, ResNet-18 ===")
target_models_cnn = {"AlexNet": alexnet_model, "ResNet-18": resnet_model}
generate_and_visualize(cnn_model, target_models_cnn, test_loader_32, epsilons, attack_regions, class_names)

# AlexNet이 공격 수행 → CNN, ResNet-18 테스트
print("=== AlexNet → CNN, ResNet-18 ===")
target_models_alexnet = {"CNN": cnn_model, "ResNet-18": resnet_model}
generate_and_visualize(alexnet_model, target_models_alexnet, test_loader_224, epsilons, attack_regions, class_names)

# ResNet-18이 공격 수행 → AlexNet, CNN 테스트
print("=== ResNet-18 → AlexNet, CNN ===")
target_models_resnet = {"CNN": cnn_model, "AlexNet": alexnet_model}
generate_and_visualize(resnet_model, target_models_resnet, test_loader_224, epsilons, attack_regions, class_names)

