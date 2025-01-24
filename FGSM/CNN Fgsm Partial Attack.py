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
            nn.Linear(128, 10)  # CIFAR-10 클래스 수
        )

    def forward(self, x):
        x = self.conv_layers(x)
        x = self.fc_layers(x)
        return x

# FGSM 공격 함수
def partial_fgsm_attack(image, epsilon, gradient, region="full"):
    perturbed_image = image.clone()
    _, h, w = image.size(1), image.size(2), image.size(3)

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

# FGSM 테스트 및 시각화 함수
def test_and_visualize_fixed_sample(model, sample_images, sample_labels, epsilon, region="full"):
    model.eval()
    correct = 0
    total_loss = 0.0

    # requires_grad 설정
    sample_images.requires_grad = True

    # 원본 예측
    outputs = model(sample_images)
    loss = criterion(outputs, sample_labels)
    total_loss += loss.item()

    # Gradient 초기화 및 Backward pass
    if sample_images.grad is not None:
        sample_images.grad.zero_()
    model.zero_grad()
    loss.backward()

    # FGSM 공격 수행
    gradient = sample_images.grad.data
    perturbed_images = partial_fgsm_attack(sample_images, epsilon, gradient, region)

    # 적대적 샘플 평가
    outputs_perturbed = model(perturbed_images)
    _, predicted = torch.max(outputs_perturbed, 1)
    correct = (predicted == sample_labels).sum().item()

    accuracy = 100 * correct / sample_labels.size(0)
    avg_loss = total_loss / sample_labels.size(0)

    # 결과 출력
    print(f"Region: {region}\tEpsilon: {epsilon:.2f}\tTest Accuracy: {accuracy:.2f}%\tAverage Loss: {avg_loss:.4f}")

    # 시각화
    visualize_attack(model, sample_images, sample_labels, epsilon, region, class_names)

    return accuracy, avg_loss

# 원본과 적대적 이미지 시각화
def visualize_attack(model, images, labels, epsilon, region, class_names):
    model.eval()

    # FGSM 공격 수행
    outputs = model(images)
    _, predicted_original = torch.max(outputs, 1)

    loss = criterion(outputs, labels)
    model.zero_grad()
    loss.backward()
    gradient = images.grad.data
    perturbed_images = partial_fgsm_attack(images, epsilon, gradient, region)

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
        img = np.transpose((img * 0.5 + 0.5), (1, 2, 0))
        plt.imshow(np.clip(img, 0, 1))
        plt.axis("off")
    plt.tight_layout()
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

# 모델 초기화 및 학습된 모델 불러오기
model = load_trained_model(SimpleCNN)

# 테스트 로더에서 첫 번째 배치 고정
data_iter = iter(test_loader)
fixed_images, fixed_labels = next(data_iter)
fixed_images, fixed_labels = fixed_images.to(device), fixed_labels.to(device)

# FGSM 공격 설정
epsilons = [0, 0.01, 0.02, 0.03, 0.04, 0.05, 0.1, 0.2]
regions = ["top_left", "top_right", "bottom_left", "bottom_right", "center", "border", "full"]

# 정확도를 저장할 딕셔너리
accuracy_dict = {region: [] for region in regions}

# 고정된 샘플에 대해 테스트 및 정확도 기록
for region in regions:
    print(f"\n=== Testing Region: {region} ===")
    for eps in epsilons:
        accuracy, avg_loss = test_and_visualize_fixed_sample(model, fixed_images, fixed_labels, eps, region)
        accuracy_dict[region].append(accuracy)

# 정확도 그래프 그리기
def plot_accuracy_graph(epsilons, accuracy_dict):
    plt.figure(figsize=(10, 6))
    for region, accuracies in accuracy_dict.items():
        plt.plot(epsilons, accuracies, marker='o', label=f'Region: {region}')
    plt.title('FGSM Attack Accuracy by Region and Epsilon')
    plt.xlabel('Epsilon')
    plt.ylabel('Accuracy (%)')
    plt.xticks(epsilons)
    plt.ylim(0, 100)
    plt.grid(True)
    plt.legend()
    plt.show()

# 정확도 그래프 출력
plot_accuracy_graph(epsilons, accuracy_dict)
