import numpy as np
import matplotlib.pyplot as plt
import torch
import random 
import time

NUMBER_OF_EXAMPLES = 5000
NUMBER_OF_EPOCHS = 100000
LR = 0.001

def train_data():
    v0 = random.uniform(5.0,25.0)
    alpha = random.uniform(0.0, np.pi/2)
    # alpha_radian = np.deg2rad(alpha)
    # alpha_deg = np.rad2deg(alpha)
    g = 9.81
    t = random.uniform(0.0,5.0)

    h = v0 * np.sin(alpha) * t - 0.5* g * t**2

    return [v0,alpha,t],[h]

def generate_data():
    examples = [train_data() for _ in range(NUMBER_OF_EXAMPLES)]

    x,y = zip(*examples)

    # x = torch.tensor(x, dtype = torch.float32)
    # y = torch.tensor(y, dtype = torch.float32)
    x = torch.tensor(x, dtype = torch.float32)
    y = torch.tensor(y, dtype = torch.float32)

    return x,y
def main():

    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    print(f"Device: {device}")

    start = time.time()
    x, y= generate_data()

    model = torch.nn.Sequential(
        torch.nn.Linear(3,32),
        torch.nn.ReLU(),
        torch.nn.Linear(32,16),
        torch.nn.ReLU(),
        torch.nn.Linear(16,8),
        torch.nn.ReLU(),
        torch.nn.Linear(8,1)
    )

    loss = torch.nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)

    

    for epoch in range(NUMBER_OF_EPOCHS):
        optimizer.zero_grad()
        pred = model(x) #forward
        current_loss = loss(pred, y)
        current_loss.backward() #backward
        optimizer.step()

        print(f"Epoch: {epoch} | Loss: {current_loss.item()}")


    test = [train_data() for _ in range(100)]

    x_test, y_test = zip(*test)

    x_test = torch.tensor(x_test, dtype = torch.float32)
    y_test = torch.tensor(y_test, dtype = torch.float32)

    with torch.no_grad():
        pred = model(x_test).squeeze()
    


    pred_y = pred.tolist()
    true_y = y_test.squeeze().tolist()

    plt.scatter(pred_y,true_y)
    plt.plot([min(true_y), max(true_y)], [min(true_y), max(true_y)], color = 'red', linewidth = 5)
    plt.grid()
    plt.show()


    # end = time.time()


    # print(f"Time:{end - start}")

if __name__ == "__main__":
    main()