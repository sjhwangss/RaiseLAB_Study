import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
from torchvision.models import resnet18
import matplotlib.pyplot as plt
import numpy as np

# Device 설정
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 데이터셋 준비 (CIFAR-10 이미지 크기를 ResNet-18에 맞게 조정)
transform = transforms.Compose([
    transforms.Resize((224, 224)),  # ResNet-18 입력 크기와 맞춤
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

test_dataset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)
test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=1, shuffle=True)

# ResNet-18 CIFAR-10용으로 수정
model = resnet18(pretrained=True)
model.fc = nn.Linear(model.fc.in_features, 10)  # CIFAR-10 클래스 수에 맞게 조정
model = model.to(device)

# CIFAR-10 클래스 이름 정의
class_names = [
    "airplane", "automobile", "bird", "cat", "deer",
    "dog", "frog", "horse", "ship", "truck"
]

# 학습된 가중치 로드
def load_trained_model(model, path='resnet18_cifar10.pth'):
    model.load_state_dict(torch.load(path))
    model.eval()
    print("학습된 모델 로드 완료")
    return model

# FGSM 공격 함수 (Partial 영역 포함)
def partial_fgsm_attack(image, epsilon, gradient, region="full"):
    perturbed_image = image.clone()
    _, h, w = image.size(1), image.size(2), image.size(3)  # 채널, 높이, 너비

    # 마스크 생성
    mask = torch.zeros_like(image).to(device)
    if region == "top_left":
        mask[:, :, :h // 2, :w // 2] = 1
    elif region == "top_right":
        mask[:, :, :h // 2, w // 2:] = 1
    elif region == "bottom_left":
        mask[:, :, h // 2:, :w // 2] = 1
    elif region == "bottom_right":
        mask[:, :, h // 2:, w // 2:] = 1
    elif region == "center":
        mask[:, :, h // 4:3 * h // 4, w // 4:3 * w // 4] = 1
    elif region == "border":
        mask[:, :, :h // 8, :] = 1
        mask[:, :, -h // 8:, :] = 1
        mask[:, :, :, :w // 8] = 1
        mask[:, :, :, -w // 8:] = 1
    elif region == "full":
        mask = torch.ones_like(image).to(device)

    # 공격 적용
    perturbed_image += epsilon * gradient.sign() * mask
    perturbed_image = torch.clamp(perturbed_image, -1, 1)
    return perturbed_image

# FGSM 테스트 함수
def test_with_fgsm(model, test_loader, epsilon, region="full"):
    model.eval()
    correct = 0
    total = 0
    total_loss = 0.0

    for images, labels in test_loader:
        images, labels = images.to(device), labels.to(device)
        images.requires_grad = True

        # Forward pass
        outputs = model(images)
        loss = criterion(outputs, labels)
        total_loss += loss.item()

        # Backward pass
        model.zero_grad()
        loss.backward()

        # FGSM 부분 공격 수행
        gradient = images.grad.data
        perturbed_images = partial_fgsm_attack(images, epsilon, gradient, region)

        # 적대적 샘플로 평가
        outputs = model(perturbed_images)
        _, predicted = torch.max(outputs, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

    accuracy = 100 * correct / total
    avg_loss = total_loss / total
    print(f"Region: {region}\tEpsilon: {epsilon:.2f}\tTest Accuracy: {accuracy:.2f}%\tAverage Loss: {avg_loss:.4f}")
    return accuracy, avg_loss

# 원본과 적대적 이미지 시각화
def visualize_attack(model, images, labels, epsilon, region, class_names):
    model.eval()
    images, labels = images.to(device), labels.to(device)
    images.requires_grad = True

    # 원본 예측
    outputs = model(images)
    _, predicted_original = torch.max(outputs, 1)

    # 공격 수행
    loss = criterion(outputs, labels)
    model.zero_grad()
    loss.backward()
    gradient = images.grad.data
    perturbed_images = partial_fgsm_attack(images, epsilon, gradient, region)

    # 적대적 예측
    outputs_perturbed = model(perturbed_images)
    _, predicted_perturbed = torch.max(outputs_perturbed, 1)

    # 시각화
    plt.figure(figsize=(10, 5))
    for i, (img, pred) in enumerate(zip([images, perturbed_images], [predicted_original, predicted_perturbed])):
        title = (
            f"Original: {class_names[labels[0]]}\nPredicted: {class_names[pred[0]]}"
            if i == 0
            else f"Perturbed (Epsilon={epsilon}, Region={region}):\nPredicted: {class_names[pred[0]]}"
        )
        plt.subplot(1, 2, i + 1)
        plt.title(title)
        img = img[0].cpu().detach().numpy()
        img = np.transpose((img * 0.5 + 0.5), (1, 2, 0))  # 정규화 해제
        plt.imshow(img)
        plt.axis("off")
    plt.tight_layout()
    plt.show()

# 손실 함수 정의
criterion = nn.CrossEntropyLoss()

# 학습된 모델 로드
model = load_trained_model(model, path='resnet18_cifar10.pth')

# FGSM 공격 강도 및 테스트 영역 설정
epsilons = [0, 0.01, 0.02, 0.03, 0.04, 0.05, 0.1, 0.2]
regions = ["top_left", "top_right", "bottom_left", "bottom_right", "center", "border", "full"]

# 결과 저장
accuracies = {region: [] for region in regions}
losses = {region: [] for region in regions}

# 테스트 로더에서 한 이미지를 고정
data_iter = iter(test_loader)
sample_images, sample_labels = next(data_iter)

# 테스트 및 시각화
for region in regions:
    print(f"\n=== Testing Region: {region} ===")
    for eps in epsilons:
        # FGSM 테스트
        acc, loss = test_with_fgsm(model, test_loader, eps, region)
        accuracies[region].append(acc)
        losses[region].append(loss)

        # 고정된 이미지로 시각화
        print(f"Visualizing for Region: {region}, Epsilon: {eps}")
        visualize_attack(model, sample_images, sample_labels, eps, region, class_names)



# 결과 시각화
for region in regions:
    plt.figure(figsize=(12, 6))
    plt.plot(epsilons, accuracies[region], label="Accuracy", marker='o')
    plt.plot(epsilons, losses[region], label="Loss", marker='x')
    plt.title(f"Region: {region}")
    plt.xlabel("Epsilon")
    plt.ylabel("Metric")
    plt.legend()
    plt.grid()
    plt.show()
