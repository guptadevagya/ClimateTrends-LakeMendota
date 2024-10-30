import sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# load data from file
def load(filename):
    try:
        data = pd.read_csv(filename)
        x = data['year'].values
        y = data['days'].values
        return x, y
    except Exception as e:
        print(f"error loading file: {filename}. check if file exists and format is right.")
        print(f"error details: {e}")
        sys.exit(1)

# normalize x values
def normalize(x):
    m = np.min(x)
    M = np.max(x)
    x_norm = (x - m) / (M - m)
    return x_norm, m, M

# add bias column to x matrix
def augment(x_norm):
    n = len(x_norm)
    X_norm = np.column_stack((x_norm, np.ones(n)))
    return X_norm

# plot year vs frozen days
def plot(x, y):
    plt.figure()
    plt.plot(x, y, color='blue')  
    plt.xlabel('Year')
    plt.ylabel('Number of Frozen Days')
    plt.savefig("data_plot.jpg")
    plt.close()

# calculate weights with closed-form solution
def closed_form(X_norm, y):
    X_t = X_norm.T
    XTX = np.dot(X_t, X_norm)
    XTX_inv = np.linalg.inv(XTX)
    XTy = np.dot(X_t, y)
    weights = np.dot(XTX_inv, XTy)
    return weights

# calculate mean squared error
def mse(w, b, X, y):
    y_pred = np.dot(X, [w, b])
    return np.mean((y_pred - y) ** 2) / 2

# gradient descent function
def gradient(X, y, learning_rate, iteration):
    n = len(y)
    w, b = 0, 0  # initializing weight and bias
    loss_hist = []
    
    print("Q5a:")
    print(np.array([w, b]))  # printing the initial weights and bias
    
    printed_rows = 1  # keep track of how many rows printed
    
    for t in range(iteration):
        y_pred = np.dot(X, [w, b])
        error = y_pred - y
        
        # calculate gradients
        w_grad = np.dot(error, X[:, 0]) / n
        b_grad = np.sum(error) / n

        # update weights and bias
        w -= learning_rate * w_grad
        b -= learning_rate * b_grad
        
        # store and print every 10 iterations 
        if (t + 1) % 10 == 0 and printed_rows < 10:
            loss_hist.append([w, b])
            print(np.array([w, b]))  # print
            printed_rows += 1  # increment

    return w, b, loss_hist

# plot loss history
def plot_loss(loss_hist, X_norm, y):
    losses = [mse(w, b, X_norm, y) for w, b in loss_hist]
    plt.figure()
    plt.plot(losses, label='loss')
    plt.xlabel('Iteration')
    plt.ylabel('Loss')
    plt.savefig("loss_plot.jpg")
    plt.close()

# predict frozen days for a given year
def predict(w, b, year, m, M):
    year_norm = (year - m) / (M - m)
    return w * year_norm + b

# get the sign of w
def sign(w):
    if w > 0:
        return '>'
    elif w < 0:
        return '<'
    else:
        return '='

# calculate when lake won't freeze
def no_freeze_year(w, b, m, M):
    x_star_norm = -b / w
    x_star = x_star_norm * (M - m) + m
    return x_star

def main():
    if len(sys.argv) != 4:
        print("usage: python3 analysis.py <filename> <learning_rate> <iterations>")
        sys.exit(1)

    filename = sys.argv[1]
    try:
        learning_rate = float(sys.argv[2])
        iteration = int(sys.argv[3])
    except ValueError:
        print("error: learning rate and iterations must be numbers.")
        sys.exit(1)

    # load data
    x, y = load(filename)

    # plot data
    plot(x, y)

    # normalize x and create matrix
    x_norm, m, M = normalize(x)
    X_norm = augment(x_norm)
    print("Q3:")
    print(X_norm)

    # closed form solution
    weights = closed_form(X_norm, y)
    w, b = weights[0], weights[1]
    print("Q4:")
    print(weights)

    # gradient descent
    w_gd, b_gd, loss_hist = gradient(X_norm, y, learning_rate, iteration)

    # plot loss vs iterations
    plot_loss(loss_hist, X_norm, y)
    print("Q5b:", learning_rate)
    print("Q5c:", iteration)

    # predict frozen days for 2023
    y_hat = predict(weights[0], weights[1], 2023, m, M)
    print("Q6:", y_hat)

    # sign of w
    symbol = sign(weights[0])
    print("Q7a:", symbol)

    # interpretation of w
    interpretation = (
        "if w > 0 it means winters are getting colder. "
        "if w < 0 it means winters are getting warmer. "
        "if w = 0 it means no trend."
    )
    print("Q7b:", interpretation)

    # year lake won't freeze
    x_star = no_freeze_year(weights[0], weights[1], m, M)
    print("Q8a:", x_star)

    analysis = (
        "the prediction is based on a linear model which may not be accurate "
        "due to external factors like climate change affecting weather patterns."
    )
    print("Q8b:", analysis)

if __name__ == "__main__":
    main()