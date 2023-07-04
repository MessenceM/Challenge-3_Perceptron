"""
Perceptron v 1.0
My first Neural Network!
It's capable of linear categorization of points separated by the function y = -x
The iterations of training are delayed in order to visualize the backpropagation
Author : Matis Messence
12/05/2023

I am on a coding adventure to learn python, data science and machine learning.
Follow my progress at
https://www.linkedin.com/in/matis-messence/
"""

from random import uniform
from tkinter import *


class Window(Tk):
    def __init__(self, width=400, height=400):
        super().__init__()
        self.width = width
        self.height = height
        self.canvas = Canvas(self, bg="light grey", height=self.height, width=self.width)
        self.canvas.pack()
        self.draw()

    def draw(self):
        self.canvas.delete("all")
        self.draw_line()
        self.draw_perceptron_line()
        self.draw_points()

    def draw_line(self):
        self.canvas.create_line(0, 0, self.width, self.height, fill="white")

    def draw_perceptron_line(self):
        # Draw the line based on the current weights
        # Formula is weights[0] * x + weights[1] * y + weights[2] = 0
        w0, w1, w2 = perceptron.weights
        x1, x2 = 0, self.width

        # Calculate the y coords in -1, +1 range
        real_y1 = ((-w2 - w0 * -1) / w1)
        real_y2 = ((-w2 - w0 * 1) / w1)

        # Map the coords to pixels coords
        y1 = ((real_y1 + 1) * self.height) / 2
        y2 = ((real_y2 + 1) * self.height) / 2
        self.canvas.create_line(x1, y1, x2, y2, fill="black", width=2)

    def draw_points(self):
        radius = 8
        thick = 2
        for index in range(len(trainer.points)):
            point = trainer.points[index]
            x = ((point.x + 1) * self.width) / 2
            y = ((point.y + 1) * self.height) / 2
            answer = perceptron.feedforward(trainer.return_input(index))
            error = perceptron.get_error(point.target, answer)
            outline = "white" if point.target == 1 else "black"
            fill = "green" if error == 0 else "red"
            self.canvas.create_oval(x - radius, y - radius, x + radius, y + radius, fill=fill, width=thick, outline=outline)

    def update_window(self):
        self.draw()
        perceptron.train()
        window.after(1000, self.update_window)


class Trainer:
    def __init__(self, nb_p):
        self.nb_points = nb_p
        self.points = []
        for i in range(self.nb_points):
            x = uniform(-1, 1)
            y = uniform(-1, 1)
            target = 1 if x >= y else -1
            point = Point(x, y, target)
            self.points.append(point)

    def return_input(self, index):
        return [self.points[index].x, self.points[index].y, self.points[index].target]


class Point:
    def __init__(self, x, y, target):
        self.x = x
        self.y = y
        self.target = target


class Perceptron:
    def __init__(self, learn_c=0.01, nb_inputs=2):
        self.nb_inputs = nb_inputs
        self.learn_c = learn_c
        self.weights = []
        self.bias = 1
        self.epoch = 0
        for j in range(self.nb_inputs):
            self.weights.append(uniform(-1, 1))
        self.weights.append(self.bias)

    def train(self):
        for epoch in range(trainer.nb_points):
            point = trainer.return_input(epoch)
            x = point[0]
            y = point[1]
            target = point[2]
            answer = self.feedforward(point)
            error = self.get_error(target, answer)
            self.epoch += 1
            for i in range(self.nb_inputs):
                self.weights[i] += self.learn_c * error * point[i]
            self.weights[-1] += self.learn_c * error * self.bias

    def feedforward(self, point):
        sum = 0
        for k in range(self.nb_inputs):
            sum += point[k] * self.weights[k]
        sum += self.bias * self.weights[-1]
        return self.activation(sum)

    def activation(self, sum):
        output = 1 if sum >= 0 else -1
        return output

    def get_error(self, target, answer):
        return target - answer


if __name__ == "__main__":
    nb_points = 200
    trainer = Trainer(nb_points)
    perceptron = Perceptron()
    window = Window()
    window.update_window()
    window.mainloop()
