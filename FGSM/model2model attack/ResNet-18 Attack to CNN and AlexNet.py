import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import numpy as np
from torchvision.models import resnet18, alexnet

# Device 설정
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# CIFAR-10 클래스 이름 정의
class_names = [
    "airplane", "automobile", "bird", "cat", "deer",
    "dog", "frog", "horse", "ship", "truck"
]

# ResNet-18을 위한 데이터 변환 (224x224 크기)
transform_resnet = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

# ResNet-18에 맞는 CIFAR-10 데이터셋 로드
test_dataset_resnet = torchvision.datasets.CIFAR10(root='./data', train=False, download=True,
                                                   transform=transform_resnet)
test_loader_resnet = torch.utils.data.DataLoader(test_dataset_resnet, batch_size=1, shuffle=True)

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

# FGSM 공격 함수
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

# 모델 로드 함수
def load_model(model_class, path):
    model = model_class().to(device)
    model.load_state_dict(torch.load(path))
    model.eval()
    return model

# ResNet-18 및 AlexNet 수정 및 로드
resnet_model = resnet18(pretrained=True)
resnet_model.fc = nn.Linear(resnet_model.fc.in_features, 10)
resnet_model = resnet_model.to(device)
resnet_model.load_state_dict(torch.load('resnet18_cifar10.pth'))
resnet_model.eval()

alexnet_model = alexnet(pretrained=True)
alexnet_model.classifier[6] = nn.Linear(4096, 10)
alexnet_model = alexnet_model.to(device)
alexnet_model.load_state_dict(torch.load('alexnet_cifar10.pth'))
alexnet_model.eval()

# CNN 모델 초기화 및 로드
cnn_model = load_model(SimpleCNN, 'cnn_cifar10.pth')

# 손실 함수 정의
criterion = nn.CrossEntropyLoss()

# 적대적 샘플 생성 및 테스트 (ResNet-18이 공격 수행)
def generate_and_visualize_resnet(model, target_models, test_loader, epsilons, regions, class_names):
    model.eval()
    for epsilon in epsilons:
        for region in regions:
            correct = {target_name: 0 for target_name in target_models.keys()}
            total = {target_name: 0 for target_name in target_models.keys()}
            images_list = []

            for images, labels in test_loader:
                images, labels = images.to(device), labels.to(device)
                images.requires_grad = True

                # ResNet-18으로 예측 및 FGSM 공격 수행
                outputs = model(images)
                loss = criterion(outputs, labels)
                model.zero_grad()
                loss.backward()
                gradient = images.grad.data

                adversarial_images = fgsm_attack(images, epsilon, gradient, region)

                # 각 모델에 맞게 크기 변환
                for target_name, target_model in target_models.items():
                    if target_name == "CNN":
                        resized_images = torch.nn.functional.interpolate(adversarial_images, size=(32, 32),
                                                                         mode='bilinear', align_corners=False)
                    else:  # AlexNet은 224x224 유지
                        resized_images = adversarial_images

                    outputs = target_model(resized_images)
                    _, predicted = torch.max(outputs, 1)
                    correct[target_name] += (predicted == labels).sum().item()
                    total[target_name] += labels.size(0)

                    if len(images_list) < 5:
                        images_list.append((images[0].cpu(), adversarial_images[0].cpu(), labels[0].cpu()))

            # 정확도 출력
            for target_name in target_models.keys():
                accuracy = 100 * correct[target_name] / total[target_name]
                print(f"Epsilon: {epsilon:.2f} | Region: {region} | Target: {target_name} | Accuracy: {accuracy:.2f}%")
            # 시각화 함수 호출 (5개 샘플)
            visualize_images(images_list, epsilon, region, class_names)

# 시각화 함수
def visualize_images(images_list, epsilon, region, class_names):
    plt.figure(figsize=(12, 6))
    for i, (original, adversarial, label) in enumerate(images_list):
        original = np.transpose((original.detach().numpy() * 0.5 + 0.5), (1, 2, 0))
        adversarial = np.transpose((adversarial.detach().numpy() * 0.5 + 0.5), (1, 2, 0))

        plt.subplot(2, 5, i + 1)
        plt.imshow(original)
        plt.title(f"Original: {class_names[label]}")
        plt.axis('off')

        plt.subplot(2, 5, i + 6)
        plt.imshow(adversarial)
        plt.title(f"Adversarial (Eps: {epsilon}, {region})")
        plt.axis('off')

    plt.tight_layout()
    plt.show()



# 실행
epsilons = [0, 0.01, 0.02, 0.03, 0.04, 0.05, 1.0, 2.0]
attack_regions = ["top_left", "top_right", "bottom_left", "bottom_right", "center", "border", "full"]
target_models = {"CNN": cnn_model, "AlexNet": alexnet_model}

generate_and_visualize_resnet(resnet_model, target_models, test_loader_resnet, epsilons, attack_regions, class_names)
