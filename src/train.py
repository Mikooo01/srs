import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import numpy as np
import random
import os
import time
import pandas as pd
from model import SimpleCNN  # ваш файл model.py с SimpleCNN

# ------------------------
# Настройка seed и device
# ------------------------
SEED = 42
torch.manual_seed(SEED)
np.random.seed(SEED)
random.seed(SEED)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

device = torch.device("cpu")  # CPU эксперимент

# ------------------------
# Параметры
# ------------------------
subset_size = 25000
val_size = 4000
train_size = subset_size - val_size  # 21000
batch_size = 64
epochs = 10
results_dir = "../results"
os.makedirs(results_dir, exist_ok=True)

activations = ["ReLU", "ELU", "LeakyReLU"]

# ------------------------
# FashionMNIST Subset с валидацией
# ------------------------
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

# Загружаем полный train датасет
full_train_set = torchvision.datasets.FashionMNIST(root='../data', train=True, download=True, transform=transform)

# Берём случайную подвыборку subset_size
subset_indices = torch.randperm(len(full_train_set))[:subset_size]
subset = torch.utils.data.Subset(full_train_set, subset_indices)

# Разделяем на train и val
train_subset, val_subset = torch.utils.data.random_split(subset, [train_size, val_size])

train_loader = torch.utils.data.DataLoader(train_subset, batch_size=batch_size, shuffle=True)
val_loader = torch.utils.data.DataLoader(val_subset, batch_size=batch_size, shuffle=False)

# Тестовый датасет (можно полный или подвыборку)
test_set = torchvision.datasets.FashionMNIST(root='../data', train=False, download=True, transform=transform)
test_loader = torch.utils.data.DataLoader(test_set, batch_size=batch_size, shuffle=False)

# ------------------------
# Функция тренировки
# ------------------------
def train_model(model, optimizer, epochs=10, scheduler=None):
    criterion = nn.CrossEntropyLoss()
    train_losses, val_losses, train_acc, val_acc = [], [], [], []

    for epoch in range(epochs):
        # --- обучение ---
        model.train()
        running_loss = 0.0
        correct, total = 0, 0

        for inputs, labels in train_loader:
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * inputs.size(0)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

        train_losses.append(running_loss/total)
        train_acc.append(correct/total)

        # --- валидация ---
        model.eval()
        running_loss_val = 0.0
        correct_val, total_val = 0, 0
        with torch.no_grad():
            for inputs, labels in val_loader:
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                running_loss_val += loss.item() * inputs.size(0)
                _, predicted = torch.max(outputs.data, 1)
                total_val += labels.size(0)
                correct_val += (predicted == labels).sum().item()

        val_losses.append(running_loss_val/total_val)
        val_acc.append(correct_val/total_val)

        if scheduler:
            scheduler.step()

        print(f"Epoch {epoch+1}/{epochs} | Train Loss: {train_losses[-1]:.4f} | "
              f"Train Acc: {train_acc[-1]:.4f} | Val Acc: {val_acc[-1]:.4f}")

    return train_losses, val_losses, train_acc, val_acc

# ------------------------
# Сравнение оптимизаторов
# ------------------------
optimizers = {
    "SGD": lambda params: optim.SGD(params, lr=0.01, momentum=0.0),
    "Momentum": lambda params: optim.SGD(params, lr=0.01, momentum=0.9),
    "Adam": lambda params: optim.Adam(params, lr=0.001)
}

all_history = []

# ------------------------
# Цикл по активациям
# ------------------------
for activation_name in activations:
    if activation_name == "ReLU":
        activation_fn = nn.ReLU
    elif activation_name == "ELU":
        activation_fn = nn.ELU
    elif activation_name == "LeakyReLU":
        activation_fn = nn.LeakyReLU
    else:
        activation_fn = nn.ReLU

    for name, opt_func in optimizers.items():
        print(f"\n--- Training with {activation_name} + {name} ---")
        model = SimpleCNN(activation=activation_fn)
        optimizer = opt_func(model.parameters())

        # LR scheduler только для SGD и Momentum
        scheduler = None
        if name in ["SGD", "Momentum"]:
            scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.5)

        start_time = time.time()
        train_losses, val_losses, train_acc, val_acc = train_model(model, optimizer, epochs=epochs, scheduler=scheduler)
        elapsed = time.time() - start_time
        print(f"Training time for {activation_name} + {name}: {elapsed:.2f} seconds")

        # Сохраняем модель
        torch.save(model.state_dict(), os.path.join(results_dir, f"model_{activation_name}_{name}.pth"))

        # Собираем историю всех эпох
        for epoch in range(epochs):
            all_history.append({
                "Activation": activation_name,
                "Optimizer": name,
                "Epoch": epoch+1,
                "Train Loss": train_losses[epoch],
                "Val Loss": val_losses[epoch],
                "Train Acc": train_acc[epoch],
                "Val Acc": val_acc[epoch],
                "Time (s)": elapsed/epochs
            })

# ------------------------
# Сохраняем всю историю в один CSV
# ------------------------
df_history = pd.DataFrame(all_history)
df_history.to_csv(os.path.join(results_dir, "training_history_fashionmnist.csv"), index=False)
print(f"\nFull training history saved to {results_dir}/training_history_fashionmnist.csv")

# ------------------------
# Построение графиков
# ------------------------
plt.figure(figsize=(12,5))

# Loss
plt.subplot(1,2,1)
for activation_name in activations:
    for name in optimizers:
        subset = df_history[(df_history["Activation"]==activation_name) & (df_history["Optimizer"]==name)]
        plt.plot(subset["Epoch"], subset["Train Loss"], label=f"{activation_name}+{name} Train")
        plt.plot(subset["Epoch"], subset["Val Loss"], '--', label=f"{activation_name}+{name} Val")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.legend(fontsize=8)
plt.title("Train/Val Loss")
plt.grid(True)

# Accuracy
plt.subplot(1,2,2)
for activation_name in activations:
    for name in optimizers:
        subset = df_history[(df_history["Activation"]==activation_name) & (df_history["Optimizer"]==name)]
        plt.plot(subset["Epoch"], subset["Train Acc"], label=f"{activation_name}+{name} Train")
        plt.plot(subset["Epoch"], subset["Val Acc"], '--', label=f"{activation_name}+{name} Val")
plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.legend(fontsize=8)
plt.title("Train/Val Accuracy")
plt.grid(True)

plt.tight_layout()
plt.savefig(os.path.join(results_dir, "loss_accuracy_fashionmnist.png"))
plt.show()
