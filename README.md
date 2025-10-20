FashionMNIST CNN Training

Бұл жоба PyTorch арқылы FashionMNIST деректерін CNN моделімен оқытуға арналған. Momentum, SGD және Adam оптимизаторларын салыстыру қарастырылған.

Құрылым
project/
├─ data/                  # Деректер автоматты түрде жүктеледі
├─ results/               # Модельдер, графиктер және CSV
├─ model.py               # SimpleCNN моделі
├─ train.py               # Оқыту скрипті
├─ requirements.txt       # Пакеттер тізімі
└─ README.md              # Жоба нұсқаулығы

Қажеттіліктер

Python >= 3.8

torch, torchvision, numpy, pandas, matplotlib

Орнату:

pip install -r requirements.txt

Оқыту
python train.py


Барлық активациялар мен оптимизатор комбинациялары оқытылады.

Модель салмақтары results/model_<activation>_<optimizer>.pth сақталады.

Training/Val Loss және Accuracy графиктері results/loss_accuracy_fashionmnist.png.

Epoch бойынша барлық нәтижелер CSV файлында сақталады: results/training_history_fashionmnist.csv.

Модель

SimpleCNN: 2 Convolution + 1 Fully Connected қабат

Активация: ReLU, ELU, LeakyReLU

Оптимизаторлар: SGD, Momentum, Adam

Epochs: 10, Batch size: 64

Пайдалану

Модельді жүктеп, тест деректері бойынша дәлдігін тексеруге болады:

import torch
from model import SimpleCNN

model = SimpleCNN()
model.load_state_dict(torch.load("results/model_ReLU_Momentum.pth"))
model.eval()