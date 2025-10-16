import tkinter as tk
import numpy as np
import pickle

CELL_SIZE = 20
GRID_SIZE = 28



def relu(x):
    return np.maximum(x,0)

def Id(x):
    return x

def softmax(x):
    x_max = np.max(x)
    e_x = np.exp(x - x_max)
    return e_x / (np.sum(e_x) + 1e-8)

hidden = int(input("Hidden layers: "))

with open(f"{hidden}layerNN.pkl", "rb") as f:
    unpickle = pickle.load(f)
    neurons = unpickle["neurons"]
    w = unpickle["weights"]
    b = unpickle["biases"]
    
layers = len(neurons)
activations = [relu] * (layers - 2) + [Id]


    

def forward(image):
    x = [None] * layers
    x[0] = image.reshape(neurons[0], 1)    
    for i in range(layers - 1):
        x[i+1] = activations[i](w[i] @ x[i] + b[i])

    print(softmax(x[-1]).resize(neurons[-1]))
    return softmax(x[-1]).argmax()




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
        self.prediction_label.config(text=str(forward(np.array(self.pixels).reshape(neurons[0]))))

    def clear(self):
        self.pixels = [[0 for _ in range(GRID_SIZE)] for _ in range(GRID_SIZE)]
        self.prediction_label.config(text="?")


        # Repaint all grid cells to white
        for i in range(GRID_SIZE):
            for j in range(GRID_SIZE):
                tag = f"cell_{i}_{j}"
                self.canvas.itemconfig(tag, fill="white")

root = tk.Tk()
root.title(f"NN Whiteboard ({hidden} hidden layers)")
wb = Whiteboard(root)
root.mainloop()
