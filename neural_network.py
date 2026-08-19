import math
import matplotlib.pyplot as plt

def sigmoid(x):
    return 1 / (1 + math.exp(-x))


def matmul(A:list,B:list):
    
    rows_a = len(A)
    cols_a = len(A[0])
    rows_b = len(B)
    cols_b = len(B[0])
    
    if cols_a != rows_b:
        raise ValueError("Matrix çarpına uygun değil")
    
    result = [[0 for _ in range(cols_b)] for _ in range(rows_a)]
    
    for i in range(rows_a):
        for j in range(cols_b):
            for k in range(cols_a):
                result[i][j] += A[i][k] *  B[k][j]
    return result


def matadd(A:list,B:list):
    result = []
    
    for i in range(len(A)):
        row=[]
        for j in range(len(A[0])):
            row.append(A[i][j]+B[i][j])
        result.append(row)
    return result



        
def apply_sigmoid(z:list):
    result = []
    
    for i in range(len(z)):
        row = []
        for j in range(len(z[i])):  
            row.append(sigmoid(z[i][j]))
        result.append(row)
    return result
        

def forward_pass(A: list, W: list, B: list):
    a = A

    for l in range(len(W)):
        z = matadd(matmul(W[l], a), B[l])
        a = apply_sigmoid(z)

    return a
 

def loss_calc(y_pred,y_true):
    return math.pow((y_pred-y_true),2)

def gradient_descent_calc(old_value, gradient, learning_rate=0.1):
    return old_value - learning_rate * gradient


def derivative_func(f,x,h=1e-5):
     return (f(x + h) - f(x - h)) / (2 * h)



A = [
    [1.0],
    [2.0]
]

W = [
    [
        [0.1, 0.2],
        [0.3, 0.1]
    ],

    [
        [0.2, -0.1],
        [0.1, 0.3]
    ]
]

B= [
    [
        [0.0],
        [0.0]
    ],

    [
        [0.0],
        [0.0]
    ]
]

y_true = 1.0

result = forward_pass(A,W,B)

y_pred = result[0][0]

print("Prediction:", y_pred)
print("Target:", y_true)
print("Loss:", loss_calc(y_pred, y_true))


def loss_for_weight(weight):
    old_weight = W[0][0][0]

    W[0][0][0] = weight

    result = forward_pass(A, W, B)
    prediction = result[0][0]
    loss = loss_calc(prediction, y_true)

    W[0][0][0] = old_weight

    return loss



weight_values = []
loss_values = []

w = -2.0

while w <= 2.0:
    loss = loss_for_weight(w)

    weight_values.append(w)
    loss_values.append(loss)

    w += 0.1


plt.plot(weight_values, loss_values)
plt.xlabel("W[0][0][0]")
plt.ylabel("Loss")
plt.title("Weight - Loss Curve")
plt.grid()
plt.show()

current_weight = W[0][0][0]

print("\nGradient descent başlıyor\n")

for i in range(30):
    gradient_derivative = derivative_func(loss_for_weight,current_weight)
    current_weight = gradient_descent_calc(current_weight,gradient_derivative,learning_rate=0.1)
    
    W[0][0][0] = current_weight
    result = forward_pass(A, W, B)
    y_pred = result[0][0]
    
    
    loss = loss_calc(
        y_pred,
        y_true
    )
    
    print(
        f"Epoch: {i:02d} | "
        f"W: {current_weight:.6f} | "
        f"Gradient: {gradient_derivative:.6f} | "
        f"Prediction: {y_pred:.6f} | "
        f"Loss: {loss:.6f}"
    )