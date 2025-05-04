# 🧠 Building a Simple Neural Network in PyTorch  

## 🔹 Step 1: Creating a Neural Network (Without Activation Function)  

Let's start with a **simple neural network** that takes an input, applies weights and bias, and gives an output. **No activation function yet!**  

```python
import torch

# Define input tensor (1 sample, 3 features)
x = torch.tensor([[2.0, 3.0, 4.0]])  # Shape: (1,3)

# Define weight tensor (3 input features → 1 output)
w = torch.tensor([[0.1, 0.2, 0.3]])  # Shape: (1,3)

# Define bias tensor
b = torch.tensor([0.5])  # Shape: (1,)

# Compute the output: y = x*w + b
y = torch.sum(x * w) + b
print("Output without activation:", y)
```

### 🎯 Task 1: Modify Weights & Bias  
🔹 Try changing the **values of weights and bias**. What happens to the output?  

#### ✅ Solution:  
When you change the weights, you are modifying the influence of each input.  
When you change the bias, you shift the output up or down.  

Try:  
```python
w = torch.tensor([[0.5, -0.3, 0.2]])  # Different weights
b = torch.tensor([1.0])  # Different bias
```
You'll see a **different output**!  

---

## 🔹 Step 2: Adding an Activation Function  

Now, let's **add an activation function** (ReLU) to introduce non-linearity.  

```python
import torch.nn.functional as F  # For activation functions

# Apply ReLU activation
y_activated = F.relu(y)
print("Output with ReLU activation:", y_activated)
```

👉 **Why do we need activation functions?**  
Without them, our network is just a linear function and **cannot model complex patterns**!  

### 🎯 Task 2: Try Different Activation Functions  
🔹 Replace `F.relu(y)` with `torch.sigmoid(y)` or `torch.tanh(y)`.  

#### ✅ Solution:  
```python
y_sigmoid = torch.sigmoid(y)
y_tanh = torch.tanh(y)

print("Sigmoid output:", y_sigmoid)
print("Tanh output:", y_tanh)
```
Each activation function **transforms the output differently**!  

---

## 🔹 Step 3: Feedforward  

A **feedforward neural network** processes inputs layer by layer to produce an output. In PyTorch, we can define it as a class:  

```python
import torch.nn as nn

class SimpleNN(nn.Module):
    def __init__(self):
        super(SimpleNN, self).__init__()
        self.linear = nn.Linear(3, 1)  # 3 input features → 1 output

    def forward(self, x):
        return self.linear(x)  # No activation yet

# Create model and sample input
model = SimpleNN()
x = torch.tensor([[2.0, 3.0, 4.0]])
output = model(x)
print("Model output:", output)
```

### 🎯 Task 3: Add an Activation Function  
🔹 Modify the network to **include a ReLU activation function** inside `forward()`.  

#### ✅ Solution:  
```python
def forward(self, x):
    return torch.relu(self.linear(x))
```
This makes the network **non-linear and more powerful**!  

---

## 🔹 Step 4: Loss Function  

A **loss function** measures how far the model's predictions are from the actual values. One common choice for regression is **Mean Squared Error (MSE)**.  

```python
# Define target (true output)
target = torch.tensor([[10.0]])

# Define Mean Squared Error Loss
loss_fn = nn.MSELoss()

# Compute loss
loss = loss_fn(output, target)
print("Loss:", loss.item())
```

### 🎯 Task 4: Try L1 Loss  
🔹 Replace `nn.MSELoss()` with `nn.L1Loss()`. What’s the difference?  

#### ✅ Solution:  
```python
loss_fn = nn.L1Loss()
loss = loss_fn(output, target)
```
- **MSE Loss** penalizes large errors more.  
- **L1 Loss** is more resistant to outliers.  

---

## 🔹 Step 5: Backpropagation & Optimizer  

The optimizer **adjusts the weights and biases** using gradient descent to minimize the loss.  

```python
# Define optimizer (Stochastic Gradient Descent)
optimizer = torch.optim.SGD(model.parameters(), lr=0.01)

# Backpropagation: Compute gradients
loss.backward()

# Update weights
optimizer.step()

# Clear gradients for next iteration
optimizer.zero_grad()
```

### 🎯 Task 5: Try the Adam Optimizer  
🔹 Change the optimizer to **Adam (`torch.optim.Adam`)** and observe the effect on training.  

#### ✅ Solution:  
```python
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
```
Adam **converges faster** and adapts the learning rate!  

---

## 🎉 Summary & Next Steps  

✅ You built a simple **neural network** from scratch!  
✅ You learned about **activation functions, loss, and optimization**.  

💚 Next, we'll **train a real dataset using PyTorch!** 🚀  

