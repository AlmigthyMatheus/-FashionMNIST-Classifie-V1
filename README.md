### **README.md**

```markdown
# 👕 Fashion Classifier V1 - Transfer Learning with ResNet18

This project is a **clothing image classifier** using **Transfer Learning** with the **ResNet18** architecture. The model is trained on the **Fashion MNIST** dataset and is optimized for **high-speed GPU inference**. 

🚀 **Version 1 (V1)** is built on **FashionMNIST**, and the upcoming **V2** will utilize the **DeepFashion** dataset for more advanced fashion classification.

## 📌 Technologies Used
- **Python 3.10+**
- **PyTorch**
- **Torchvision**
- **CUDA (GPU)**
- **Matplotlib**

## 📂 Project Structure
```
📁 Fashion-Classifier-V1
│── 📂 data/               # Dataset folder (auto-downloaded)
│── 📂 models/             # Final trained model
│── 📂 checkpoints/        # Model checkpoints (every 10 epochs)
│── 📂 tests/              # Training graphs and other results
│── 📝 README.md           # Project documentation
│── 📜 train_fashion.py    # Model training script
│── 📜 requirements.txt    # Required dependencies
```

## 🚀 How to Train the Model
1. **Clone the repository**
   ```bash
   git clone https://github.com/your-username/Fashion-Classifier-V1.git
   cd Fashion-Classifier-V1
   ```

2. **Create and activate a virtual environment (optional)**
   ```bash
   python -m venv env
   source env/bin/activate  # Linux/macOS
   env\Scripts\activate     # Windows
   ```

3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

4. **Run the training script**
   ```bash
   python train_fashion.py
   ```

5. **The trained model and checkpoints will be saved automatically in the `checkpoints/` folder.**

## 🎯 Results
- The model classifies clothing images into **10 categories**:
  `T-shirt/top, Trouser, Pullover, Dress, Coat, Sandal, Shirt, Sneaker, Bag, Ankle boot`.
- **Transfer Learning** with **ResNet18**, optimizing the final layer for **10 classes**.
- **GPU acceleration (CUDA) + Mixed Precision Training** for **faster processing**.

## 🛠️ Next Steps (V2)
The next version of this project (**Fashion Classifier V2**) will be trained on the **DeepFashion** dataset, allowing for **more advanced and precise clothing classification**.

---

## 📜 License
This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for more details.

---

## ✨ Author
Project developed by **Matheus**. Feel free to reach out via GitHub!

---
