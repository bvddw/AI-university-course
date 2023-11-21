from tkinter import *
from PIL import ImageGrab
import numpy as np
from main import NeuralNetwork as nn
from training import test_fcn


def draw(event):
    x1, y1 = (event.x - 20), (event.y - 20)
    x2, y2 = (event.x + 20), (event.y + 20)
    c.create_oval(x1, y1, x2, y2, fill="black", outline="black")


def clear(event):
    c.delete("all")
    res.set(value='')


def recognize():
    x1 = root.winfo_rootx() + c.winfo_x()
    y1 = root.winfo_rootx() + c.winfo_x()
    x2 = x1 + c.winfo_width()
    y2 = y1 + c.winfo_height()

    img = ImageGrab.grab(bbox=(x1 + 2, y1 + 2, x2 - 4, y2 - 4))
    img = img.resize((28, 28)).convert('L')
    inputs = np.asfarray(img).reshape(1, 784)
    inputs = (255.0 - inputs) / 255.0 * 0.99 + 0.01
    outputs = network.query(inputs)
    label = np.argmax(outputs)
    res.set(value=str(label))
    print(outputs)


def train():
    res.set(value='')
    x1 = root.winfo_rootx() + c.winfo_x()
    y1 = root.winfo_rooty() + c.winfo_y()
    x2 = x1 + c.winfo_width()
    y2 = y1 + c.winfo_height()
    img = ImageGrab.grab(bbox=(x1 + 2, y1 + 2, x2 - 4, y2 - 4))
    img = img.resize((28, 28)).convert("L")
    inputs = np.asfarray(img).reshape(1, 784)
    inputs = (255.0 - inputs) / 255.0 * 0.99 + 0.01
    targets = np.zeros(10) + 0.01
    mark = int(cor_answer.get())
    if mark not in range(10):
        return None
    targets[mark] = 0.99
    network.train(inputs_list=inputs, targets_list=targets)
    network.save_weights()
    pass


network = nn()
network.load_weights("weights_input_hidden.npy", "weights_hidden_output.npy")
test_fcn(network)
root = Tk()
root.title("Recognition Digit")
root.geometry("1150x700+400+250")
root.resizable(FALSE, FALSE)
c = Canvas(root, bg="white", bd=1, highlightbackground="black")
c.place(x=50, y=50, width=500, height=500)
c.bind("<B1-Motion>", draw)
c.bind("<Button-3>", clear)
res = StringVar(root, value='')
Entry(root, borderwidth=3, textvariable=res, justify="center",
      font=("Helvetica", 300), state=DISABLED).place(x=600, y=50, width=500, height=500)
Button(root, bg="#E1E1E1", font=("BankGothic Md Bt", 38), text="Recognize",
       command=recognize).place(x=50, y=583, width=500, height=80)
Button(root, bg="#E1E1E1", font=("BankGothic Md Bt", 38), text="Train",
       command=train).place(x=600, y=583, width=300, height=80)

cor_answer = StringVar(root, value='')
Entry(root, borderwidth=3, textvariable=cor_answer, justify="center", font=("Helvetica", 20)).place(x=910, y=583, width=190, height=80)
root.mainloop()

