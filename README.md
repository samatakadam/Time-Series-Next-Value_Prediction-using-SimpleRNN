# 📈 Value Prediction using Simple RNN

## 🧠 Project Overview

This project demonstrates how a **Simple Recurrent Neural Network (RNN)** can be used to **predict the next value in a numerical sequence**.
The model is trained on a sequence of numbers and learns the underlying pattern using a **sliding window approach**.

---

## 👩‍💻 Author

**Name:** Samata
**Date:** 2026-01-06

---

## 🛠 Technologies Used

* **Python 3**
* **NumPy**
* **TensorFlow / Keras**
* **SimpleRNN**

---

## 📂 Project Structure

```
ValuePrediction1/
│
├── ValuePrediction1.py   # Main Python program
├── README.md             # Project documentation
```

---

## 📊 Dataset Description

* The dataset consists of a **sequence of numbers from 0 to 199**
* Data is **normalized to the range [0, 1]** for better training performance
* A **sliding window of size 5** is used to predict the next value

---

## ⚙️ How the Model Works

1. Generate a numerical sequence
2. Normalize the data
3. Create input-output pairs using a sliding window
4. Train a **SimpleRNN** model
5. Predict the next number based on user input

---

## 🧪 Model Architecture

* **Input Layer:** 5 time steps, 1 feature
* **Hidden Layer:** SimpleRNN with 32 neurons
* **Output Layer:** Dense layer with 1 neuron
* **Activation Function:** Tanh
* **Loss Function:** Mean Squared Error (MSE)
* **Optimizer:** Adam

---

## ▶️ How to Run the Project

### 1️⃣ Install Required Libraries

```bash
pip install numpy tensorflow
```

### 2️⃣ Run the Program

```bash
python ValuePrediction1.py
```

---

## 🧑‍💻 User Interaction

After training, the program asks the user to enter **5 numbers**:

```
Enter 5 numbers separated by space (or type 'quit'):
```

### Example Input:

```
10 11 12 13 14
```

### Output:

```
Input: [10, 11, 12, 13, 14] → Predicted next value: 15
```

---

## ✅ Features

* Clean and organized code structure
* Real-time predictions
* Input validation and error handling
* Reproducible results using fixed random seeds

---

## 🚀 Future Enhancements

* Replace SimpleRNN with **LSTM or GRU**
* Predict multiple future values
* Visualize training loss using graphs
* Use real-world time-series datasets

---




