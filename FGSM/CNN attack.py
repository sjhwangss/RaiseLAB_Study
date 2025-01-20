import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import numpy as np

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 데이터셋 준비
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])
test_dataset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)
test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=1, shuffle=True)  # 배치 크기를 1로 설정

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

# 저장된 모델 로드
def load_trained_model(model_class, path='cnn_cifar10.pth'):
    model = model_class().to(device)
    model.load_state_dict(torch.load(path))
    model.eval()
    print("학습된 모델 로드 완료")
    return model

# FGSM 공격 함수
def fgsm_attack(image, epsilon, gradient):
    perturbed_image = image + epsilon * gradient.sign()
    perturbed_image = torch.clamp(perturbed_image, -1, 1)
    return perturbed_image

# FGSM 테스트 함수
def test_with_fgsm(model, test_loader, epsilon):
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

        # Backward pass to 계산 그래디언트
        model.zero_grad()
        loss.backward()

        # FGSM 공격 수행
        gradient = images.grad.data
        perturbed_images = fgsm_attack(images, epsilon, gradient)

        # 적대적 샘플로 평가
        outputs = model(perturbed_images)
        _, predicted = torch.max(outputs, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

    accuracy = 100 * correct / total
    avg_loss = total_loss / total
    print(f"Epsilon: {epsilon}\tTest Accuracy: {accuracy:.2f}%\tAverage Loss: {avg_loss:.4f}")
    return accuracy, avg_loss


# 원본과 적대적 이미지 시각화
def visualize_attack(model, test_loader, epsilon):
    model.eval()
    data_iter = iter(test_loader)
    images, labels = next(data_iter)
    images, labels = images.to(device), labels.to(device)
    images.requires_grad = True

    # 원본 예측
    outputs = model(images)
    _, predicted = torch.max(outputs, 1)

    # 공격 수행
    loss = criterion(outputs, labels)
    model.zero_grad()
    loss.backward()
    gradient = images.grad.data
    perturbed_images = fgsm_attack(images, epsilon, gradient)

    # 원본과 공격 당한 이미지 시각화
    plt.figure(figsize=(8, 4))
    for i in range(2):
        img = images if i == 0 else perturbed_images
        title = "Original" if i == 0 else f"Perturbed (Epsilon={epsilon})"
        plt.subplot(1, 2, i+1)
        plt.title(title)
        img = img[0].cpu().detach().numpy()
        img = np.transpose((img * 0.5 + 0.5), (1, 2, 0))  # 정규화 해제
        plt.imshow(img)
        plt.axis('off')
    plt.show()


# 손실 함수 정의
criterion = nn.CrossEntropyLoss()


# 저장된 모델 불러오기
model = load_trained_model(SimpleCNN)


# FGSM 공격 강도 설정 및 테스트
epsilons = [0, 0.01, 0.1, 0.2, 0.3]
accuracies, losses = [], []

for eps in epsilons:
    acc, loss = test_with_fgsm(model, test_loader, eps)
    accuracies.append(acc)
    losses.append(loss)

    # 첫 번째 epsilon 값에 대해 시각화
    if eps == epsilons[0]:
        visualize_attack(model, test_loader, eps)

# FGSM 공격 강도에 따른 정확도 및 손실 시각화
plt.figure(figsize=(12, 5))
plt.subplot(1, 2, 1)
plt.plot(epsilons, accuracies, marker='o')
plt.title("Accuracy vs Epsilon")
plt.xlabel("Epsilon")
plt.ylabel("Accuracy (%)")

plt.subplot(1, 2, 2)
plt.plot(epsilons, losses, marker='o', color='red')
plt.title("Loss vs Epsilon")
plt.xlabel("Epsilon")
plt.ylabel("Average Loss")

plt.tight_layout()
plt.show()
