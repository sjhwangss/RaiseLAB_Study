import matplotlib.pyplot as plt
import torchvision.transforms as transforms
from torchvision.datasets import CIFAR10
from PIL import Image
import torch

# CIFAR-10 샘플 이미지를 시각화하기 위한 각 전처리 파이프라인 (정규화 없이)
transform_display = {
    'baseline': transforms.Compose([
        transforms.ToTensor(),         # PIL Image -> Tensor
        transforms.ToPILImage()          # Tensor -> PIL Image (시각화용)
    ]),
    'augment': transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.ToPILImage()
    ]),
    'autoaugment': transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.AutoAugment(policy=transforms.AutoAugmentPolicy.CIFAR10),
        transforms.ToTensor(),
        transforms.ToPILImage()
    ]),
    'randaugment': transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.RandAugment(),
        transforms.ToTensor(),
        transforms.ToPILImage()
    ]),
    'colorjitter': transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
        transforms.ToTensor(),
        transforms.ToPILImage()
    ])
}

# CIFAR-10 데이터셋에서 한 샘플 이미지 로드 (train set)
dataset = CIFAR10(root='./data', train=True, download=True)
sample_image, label = dataset[0]  # sample_image는 PIL Image 형식입니다.

# 각 전처리 방식으로 변환한 이미지를 한 줄에 나란히 출력
plt.figure(figsize=(15, 3))
for i, (name, transform) in enumerate(transform_display.items()):
    # 각 방식은 랜덤 요소를 포함하므로 실행할 때마다 다른 결과가 나올 수 있습니다.
    transformed_image = transform(sample_image)
    plt.subplot(1, len(transform_display), i + 1)
    plt.imshow(transformed_image)
    plt.title(name)
    plt.axis('off')
plt.suptitle("Examples of applying each preprocessing method", fontsize=16)
plt.show()
