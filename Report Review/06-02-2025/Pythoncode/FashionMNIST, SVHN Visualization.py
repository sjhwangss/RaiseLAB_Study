import torch
import torchvision.datasets as datasets
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import numpy as np

# FashionMNIST 클래스 이름 정의
fashion_mnist_labels = [
    "T-shirt/top", "Trouser", "Pullover", "Dress", "Coat",
    "Sandal", "Shirt", "Sneaker", "Bag", "Ankle boot"
]

# 데이터 변환 (FashionMNIST: 1채널 -> 3채널 변환)
transform_fashion_mnist = transforms.Compose([
    transforms.Grayscale(num_output_channels=3),
    transforms.ToTensor()
])

transform_svhn = transforms.ToTensor()  # SVHN은 이미 3채널

# FashionMNIST 및 SVHN 데이터셋 로드
fashion_mnist = datasets.FashionMNIST(root="./data", train=True, download=True, transform=transform_fashion_mnist)
svhn = datasets.SVHN(root="./data", split="train", download=True, transform=transform_svhn)

# 각 클래스별 샘플 저장을 위한 딕셔너리
fashion_mnist_samples = {i: None for i in range(10)}
svhn_samples = {i: None for i in range(10)}

# FashionMNIST에서 클래스별 샘플 찾기
for img, label in fashion_mnist:
    if fashion_mnist_samples[label] is None:
        fashion_mnist_samples[label] = img
    if all(v is not None for v in fashion_mnist_samples.values()):
        break

# SVHN에서 클래스별 샘플 찾기
for img, label in zip(svhn.data, svhn.labels):
    label = int(label)  # SVHN은 numpy 배열로 저장됨
    if svhn_samples[label] is None:
        svhn_samples[label] = torch.tensor(img).permute(2, 0, 1) / 255.0  # 정규화
    if all(v is not None for v in svhn_samples.values()):
        break

# 시각화 함수 정의
def plot_images(dataset_name, samples, labels):
    fig, axes = plt.subplots(1, 10, figsize=(15, 2))
    fig.suptitle(dataset_name, fontsize=16)

    for i, ax in enumerate(axes):
        img = samples[i].permute(1, 2, 0).numpy()  # (C, H, W) → (H, W, C)

        # FashionMNIST는 흑백, SVHN은 컬러 처리
        if dataset_name == "FashionMNIST":
            ax.imshow(img.squeeze(), cmap="gray")  # 흑백 이미지
        else:
            ax.imshow(img)  # SVHN은 컬러 이미지

        ax.set_title(labels[i] if labels else str(i))
        ax.axis("off")

    plt.show()

# FashionMNIST 및 SVHN 클래스별 샘플 이미지 출력
plot_images("FashionMNIST", fashion_mnist_samples, fashion_mnist_labels)
plot_images("SVHN", svhn_samples, [str(i) for i in range(10)])
