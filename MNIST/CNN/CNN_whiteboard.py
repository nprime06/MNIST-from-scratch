import tkinter as tk
import numpy as np
from numpy.lib.stride_tricks import sliding_window_view
import pickle

CELL_SIZE = 20
GRID_SIZE = 28

def relu(x):
    return np.maximum(x,0)

def softmax(x):
    x_max = np.max(x)
    e_x = np.exp(x - x_max)
    return e_x / (np.sum(e_x) + 1e-8)

with open("MNIST_CNN_wb.pkl", "rb") as f:
    unpickle = pickle.load(f)
    w = unpickle["weights"]
    b = unpickle["biases"]

def conv(in_channel, kernel, bias): # vectorized
    # stride: 1
    # padding: 1 (zeros)
    # kernel size 3 x 3 (preserves in_channel 2d shape)

    N, C_in, h, w = in_channel.shape
    C_out, _, kh, kw = kernel.shape
    # bias.shape: (C_out,)

    padded_in_channel = np.pad(in_channel, ((0, 0), (0, 0), (1, 1), (1, 1)), mode = 'constant')
    windowed_in_channel = sliding_window_view(padded_in_channel, window_shape = (kh, kw), axis = (2, 3)).transpose(0, 2, 3, 1, 4, 5).reshape(N * h * w, C_in * kh * kw)
    flat_kernel = kernel.reshape(C_out, C_in * kh * kw) # note: array vals still copied by reference (OK here not edited)
    out_channel = (flat_kernel @ windowed_in_channel.T).reshape(C_out, N, h, w).transpose(1, 0, 2, 3) + bias.reshape(1, C_out, 1, 1)

    return out_channel

def max_pool(in_channel): # vectorized
    # pool: (2, 2)
    # stride: 2

    N, C_in, h, w = in_channel.shape
    oh, ow = h // 2, w // 2
    blocked_in_channel = in_channel.reshape(N, C_in, oh, 2, ow, 2).transpose(0, 1, 2, 4, 3, 5) # note: array vals still copied by reference (OK here not edited)
    out_channel = np.max(blocked_in_channel, axis = (4, 5))
    mask = (blocked_in_channel == out_channel.reshape(N, C_in, oh, ow, 1, 1)).astype(int).transpose(0, 1, 2, 4, 3, 5).reshape(N, C_in, h, w)

    return out_channel, mask


def forward(image):
    # image.shape: (784)

    x = [None] * 15

    x[0] = image.reshape(1, 1, 28, 28) # image 
    x[1] = conv(x[0], w[0], b[0])
    x[2] = relu(x[1])
    x[3] = conv(x[2], w[1], b[1])
    x[4] = relu(x[3])
    x[5], _ = max_pool(x[4])
    x[6] = conv(x[5], w[2], b[2])
    x[7] = relu(x[6])
    x[8] = conv(x[7], w[3], b[3])
    x[9] = relu(x[8])
    x[10], _ = max_pool(x[9])
    x[11] = x[10].reshape(1, -1)
    x[12] = x[11] @ w[4].T + b[4].reshape(1, -1)
    x[13] = relu(x[12])
    x[14] = x[13] @ w[5].T + b[5].reshape(1, -1)# last layer not activated
    
    print(str(softmax(x[-1][0]).resize(10)))
    return softmax(x[-1][0]).argmax()




class Whiteboard:
    def __init__(self, root):
        self.frame = tk.Frame(root)
        self.frame.pack()

        self.canvas = tk.Canvas(self.frame, width=GRID_SIZE*CELL_SIZE, height=GRID_SIZE*CELL_SIZE, bg="white")
        self.canvas.grid(row=0, column=0)

        # Prediction digit display
        self.prediction_label = tk.Label(self.frame, text="?", font=("Helvetica", 48), width=2, height=1)
        self.prediction_label.grid(row=0, column=1, padx=20)
        
        self.pixels = [[0 for _ in range(GRID_SIZE)] for _ in range(GRID_SIZE)]  # 2D array to store pixel values

        # Draw the grid
        for i in range(GRID_SIZE):
            for j in range(GRID_SIZE):
                x1 = j * CELL_SIZE
                y1 = i * CELL_SIZE
                x2 = x1 + CELL_SIZE
                y2 = y1 + CELL_SIZE
                self.canvas.create_rectangle(x1, y1, x2, y2, outline="lightgray", fill="white", tags=f"cell_{i}_{j}")

        # Bind mouse events
        self.canvas.bind("<B1-Motion>", self.paint)
        self.canvas.bind("<Button-1>", self.paint)

        button_frame = tk.Frame(root)
        button_frame.pack(pady=5)

        tk.Button(button_frame, text="Export Pixels", command=self.export).pack(side=tk.LEFT, padx=5)
        tk.Button(button_frame, text="Clear", command=self.clear).pack(side=tk.LEFT, padx=5)

    def paint(self, event): 
        col = event.x // CELL_SIZE
        row = event.y // CELL_SIZE
        if 0 <= row < GRID_SIZE and 0 <= col < GRID_SIZE:
            tag = f"cell_{row}_{col}"
            self.canvas.itemconfig(tag, fill="black")
            self.pixels[row][col] = 1  # mark as "inked"

    def export(self):
        self.prediction_label.config(text=str(forward(np.array(self.pixels).reshape(784)))) # need to normalize / standardize

    def clear(self):
        self.pixels = [[0 for _ in range(GRID_SIZE)] for _ in range(GRID_SIZE)]
        self.prediction_label.config(text="?")


        # Repaint all grid cells to white
        for i in range(GRID_SIZE):
            for j in range(GRID_SIZE):
                tag = f"cell_{i}_{j}"
                self.canvas.itemconfig(tag, fill="white")

root = tk.Tk()
root.title(f"CNN Whiteboard")
wb = Whiteboard(root)
root.mainloop()
