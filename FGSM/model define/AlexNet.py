import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torchvision.models import alexnet

# 데이터셋 준비
transform = transforms.Compose([
    transforms.Resize(224),  # AlexNet은 입력 크기가 224x224여야 함
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

train_dataset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
test_dataset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)

train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=128, shuffle=True)
test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=128, shuffle=False)

# AlexNet 모델 정의
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = alexnet(pretrained=True)  # 사전 학습된 가중치 사용
model.classifier[6] = nn.Linear(4096, 10)  # CIFAR-10 클래스에 맞게 출력 크기 변경
model = model.to(device)

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# 학습 함수
def train_model(model, train_loader, criterion, optimizer, num_epochs=10):
    model.train()
    for epoch in range(num_epochs):
        running_loss = 0.0
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)

            # Forward pass
            outputs = model(images)
            loss = criterion(outputs, labels)

            # Backward pass and optimization
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

        print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {running_loss/len(train_loader):.4f}")

    # 모델 저장
    torch.save(model.state_dict(), 'alexnet_cifar10.pth')
    print("AlexNet 모델 저장 완료: alexnet_cifar10.pth")

# 학습 실행
train_model(model, train_loader, criterion, optimizer)

# 모델 로드
def load_trained_model(model, path='alexnet_cifar10.pth'):
    model.load_state_dict(torch.load(path))
    model.to(device)
    print("AlexNet 학습된 모델 로드 완료")
    return model

model = load_trained_model(model)

# 테스트 정확도 평가
def evaluate_model(model, test_loader):
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    print(f"Test Accuracy: {100 * correct / total:.2f}%")

evaluate_model(model, test_loader)

