import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import numpy as np

# Device 설정
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 데이터셋 준비
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

test_dataset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)
test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=1, shuffle=True)

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
    perturbed_image = image.clone()
    c, h, w = image.size(1), image.size(2), image.size(3)  # 채널, 높이, 너비

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

# 테스트 함수
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

        # Backward pass
        model.zero_grad()
        loss.backward()

        # FGSM 공격 수행
        gradient = images.grad.data
        perturbed_images = fgsm_attack(images, epsilon, gradient, region)

        # 적대적 샘플로 평가
        outputs = model(perturbed_images)
        _, predicted = torch.max(outputs, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

        total_loss += loss.item()

    accuracy = 100 * correct / total
    avg_loss = total_loss / total
    print(f"Region: {region} | Epsilon: {epsilon} | Accuracy: {accuracy:.2f}% | Loss: {avg_loss:.4f}")
    return accuracy, avg_loss

# 시각화 함수
def visualize_attack(model, image, label, epsilon, region, class_names):
    model.eval()
    image, label = image.to(device), label.to(device)
    image.requires_grad = True

    # 원본 예측
    output = model(image)
    _, pred_original = torch.max(output, 1)

    # 공격 수행
    loss = criterion(output, label)
    model.zero_grad()
    loss.backward()
    gradient = image.grad.data
    perturbed_image = fgsm_attack(image, epsilon, gradient, region)

    # 적대적 예측
    output_perturbed = model(perturbed_image)
    _, pred_perturbed = torch.max(output_perturbed, 1)

    # 시각화
    plt.figure(figsize=(10, 5))
    for i, img in enumerate([image, perturbed_image]):
        title = (
            f"Original: {class_names[label.item()]}, Pred: {class_names[pred_original.item()]}"
            if i == 0 else
            f"Perturbed (Epsilon={epsilon}): Pred: {class_names[pred_perturbed.item()]}"
        )
        plt.subplot(1, 2, i + 1)
        plt.title(title)
        img = img[0].cpu().detach().numpy()
        img = np.transpose((img * 0.5 + 0.5), (1, 2, 0))
        plt.imshow(img)
        plt.axis("off")
    plt.show()

# 손실 함수 정의
criterion = nn.CrossEntropyLoss()

# 저장된 모델 불러오기
def load_trained_model(model_class, path='cnn_cifar10.pth'):
    model = model_class().to(device)
    model.load_state_dict(torch.load(path))
    model.eval()
    print("Model loaded successfully")
    return model

# CIFAR-10 클래스 이름 정의
class_names = [
    "airplane", "automobile", "bird", "cat", "deer",
    "dog", "frog", "horse", "ship", "truck"
]

# 저장된 모델 불러오기
model = load_trained_model(SimpleCNN)

# FGSM 공격 수행
regions = ["top_left", "top_right", "bottom_left", "bottom_right", "center", "border", "full"]
epsilons = [0, 0.01, 0.02, 0.03, 0.04, 0.05, 0.1, 0.2]

accuracies = {region: [] for region in regions}
losses = {region: [] for region in regions}

for region in regions:
    print(f"Testing region: {region}")
    for eps in epsilons:
        acc, loss = test_with_fgsm(model, test_loader, eps, region)
        accuracies[region].append(acc)
        losses[region].append(loss)

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
