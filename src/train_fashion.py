import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from torchvision import models
from torchvision.models import ResNet18_Weights
import matplotlib.pyplot as plt
import os

# 🚀 Configuração para máxima performance
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.backends.cudnn.benchmark = True  # Aceleração cuDNN
torch.backends.cuda.matmul.allow_tf32 = True  # Permitir TF32 para velocidade extra

# Criar diretórios
os.makedirs("models", exist_ok=True)
os.makedirs("testes", exist_ok=True)
os.makedirs("checkpoints", exist_ok=True)

# 📌 Transformação para Fashion MNIST
transform = transforms.Compose([
    transforms.Resize((64, 64)),  # Reduz para 64x64 para menos processamento
    transforms.Grayscale(num_output_channels=3),
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

# 📥 Carregar Fashion MNIST
train_dataset = torchvision.datasets.FashionMNIST(root="./data", train=True, transform=transform, download=True)
test_dataset = torchvision.datasets.FashionMNIST(root="./data", train=False, transform=transform, download=True)

# Criar DataLoaders com ajustes para evitar erros no Windows
train_loader = DataLoader(train_dataset, batch_size=128, shuffle=True, num_workers=4, pin_memory=True)
test_loader = DataLoader(test_dataset, batch_size=128, shuffle=False, num_workers=4, pin_memory=True)

# Classes do Fashion MNIST
classes = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat',
           'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']

# 📌 Criando o modelo usando ResNet18 pré-treinada
class FashionMNISTModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = models.resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)
        self.model.fc = nn.Linear(self.model.fc.in_features, 10)  # Ajustar para 10 classes

    def forward(self, x):
        return self.model(x)

# Inicializar o modelo na GPU
model = FashionMNISTModel().to(device)

# 🚀 Configurar otimizador AdamW e função de perda
criterion = nn.CrossEntropyLoss()
optimizer = optim.AdamW(model.parameters(), lr=0.0005, betas=(0.9, 0.999))

# 📌 Treinamento com Mixed Precision para acelerar
losses = []

def save_checkpoint(model, epoch):
    checkpoint_path = f"checkpoints/model_epoch_{epoch}.pth"
    torch.save(model.state_dict(), checkpoint_path)
    print(f"✅ Checkpoint salvo: {checkpoint_path}")

def plot_loss_curve(losses):
    plt.figure(figsize=(8,5))
    plt.plot(range(1, len(losses) + 1), losses, marker='o', linestyle='-', color='b', label='Training Loss')
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Loss Curve During Training")
    plt.legend()
    plt.grid()
    plt.savefig("testes/training_loss_curve.png")  # Salvar gráfico
    plt.show()

def train_model(model, train_loader, criterion, optimizer, epochs=100):
    model.train()
    print(f"🚀 Training on {device} with Mixed Precision for {epochs} epochs...")

    scaler = torch.amp.GradScaler()  # Usando a versão mais recente do GradScaler

    for epoch in range(epochs):
        running_loss = 0.0
        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device, non_blocking=True), labels.to(device, non_blocking=True)

            optimizer.zero_grad()
            with torch.amp.autocast(device_type="cuda"):
                outputs = model(inputs)
                loss = criterion(outputs, labels)

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            running_loss += loss.item()
        losses.append(running_loss / len(train_loader))
        print(f"✅ Epoch {epoch+1} - Loss: {running_loss/len(train_loader):.4f}")

        if epoch % 10 == 0:  # Salvar checkpoint a cada 10 épocas
            save_checkpoint(model, epoch)

    plot_loss_curve(losses)

# 📌 Garantir que o código só rode se chamado diretamente
if __name__ == "__main__":
    import torch.multiprocessing as mp
    mp.set_start_method('spawn', force=True)  # Corrigir multiprocessing no Windows

    print(f"🔥 Using GPU: {torch.cuda.get_device_name(0)} (CUDA {torch.version.cuda})")

    # Iniciar o treinamento
    train_model(model, train_loader, criterion, optimizer)
