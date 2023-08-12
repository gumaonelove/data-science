import torch
import yaml

from abc import ABC
from typing import List


class Task(ABC):
    def solve():
        """
        Function to implement your solution, write here
        """

    def evaluate():
        """
        Function to evaluate your solution
        """


class Task1(Task):
    """
        Calculate, using PyTorch, the sum of the elements of the range from 0 to 10000.
    """

    def __init__(self) -> None:
        self.task_name = "task1"

    def solve(self):
        # write your solution here
        return torch.sum(torch.tensor([range(0, 10_000)]))

    def evaluate(self):
        solution = self.solve()

        return {self.task_name: {"answer": solution.item()}}


class Task2(Task):
    """
        Solve optimization problem: find the minimum of the function f(x) = ||Ax^2 + Bx + C||^2, where
        - x is vector of size 8
        - A is identity matrix of size 8x8
        - B is matrix of size 8x8, where each element is 0
        - C is vector of size 8, where each element is -1

        Use PyTorch and autograd function. Relative error will be less than 1e-3
        
        Solution here is x, converted to the list(see submission.yaml).
    """

    def __init__(self) -> None:
        self.task_name = "task2"
        self.A = torch.eye(8)
        self.B = torch.zeros((8, 8))
        self.C = -torch.ones(8)
        self.learning_rate = 0.01
        self.epochs = 10000
        self.rel_err_threshold = 1e-3

    def forward(self, X):
        return torch.norm(torch.matmul(self.A, X ** 2) + torch.matmul(self.B, X) + self.C) ** 2

    def solve(self):
        # write your solution here
        X = torch.randn(8, requires_grad=True)
        for i in range(self.epochs):
            fx = self.forward(X)
            fx.backward()

            # Update x using its gradient and the learning rate
            with torch.no_grad():
                X -= self.learning_rate * X.grad

            # Zero out the gradient for the next iteration
            X.grad.zero_()

            # Check for convergence
            with torch.no_grad():
                if i % 100 == 0:
                    fx_new = self.forward(X)
                    rel_err = torch.abs(fx_new - fx) / fx
                    if rel_err < self.rel_err_threshold:
                        break

        return X

    def evaluate(self):
        solution = self.solve()

        return {self.task_name: {"answer": solution.tolist()}}


class Task3(Task):
    """
        Solve optimization problem: find the optimal parameters of the linear regression model, using PyTorch.
        train_X = [[0, 0], [1, 0], [0, 1], [1, 1]],
        train_y = [1.0412461757659912, 0.5224423408508301, 0.5145719051361084, 0.052878238260746]

        text_X = [[0, -1], [-1, 0]]

        User PyTorch. Relative error will be less than 1e-1
        
        Solution here is test_y, calculated from test_X, converted to the list(see submission.yaml).
    """

    def __init__(self) -> None:
        self.task_name = "task3"

        self.train_X = torch.tensor([[0, 0], [1, 0], [0, 1], [1, 1]], dtype=torch.float)
        self.test_X = torch.tensor([[0, -1], [-1, 0]], dtype=torch.float)
        self.train_y = torch.tensor(
            [1.0412461757659912, 0.5224423408508301, 0.5145719051361084, 0.052878238260746],
            dtype=torch.float
        ).unsqueeze(1)

        self.w = torch.randn((2, 1), requires_grad=True)
        self.b = torch.randn(1, requires_grad=True)

        self.lr = 0.01
        self.error = 1e-1
        self.num_epochs = 1_000

    def linear_regression(self, x, w, b):
        return torch.matmul(x, w) + b

    def mse(self, true, predicted):
        return ((true - predicted) ** 2).sum() / true.numel()

    def fit(self):
        for epoch in range(self.num_epochs):
            outputs = self.linear_regression(self.train_X, self.w, self.b)
            loss = self.mse(outputs, self.train_y)
            loss.backward()

            with torch.no_grad():
                self.w -= self.lr * self.w.grad
                self.b -= self.lr * self.b.grad
                self.w.grad.zero_()
                self.b.grad.zero_()

    def solve(self):
        # write your solution here
        self.fit()
        with torch.no_grad():
            test_y = self.linear_regression(self.train_X, self.w, self.b)
            test_y = test_y.squeeze(1)

        return test_y

    def evaluate(self):
        solution = self.solve()

        return {self.task_name: {"answer": solution.tolist()}}


class HW(object):
    def __init__(self, list_of_tasks: List[Task]):
        self.tasks = list_of_tasks
        self.hw_name = "dl_lesson_1_checker_hw"

    def evaluate(self):
        aggregated_tasks = []

        for task in self.tasks:
            aggregated_tasks.append(task.evaluate())

        aggregated_tasks = {"tasks": aggregated_tasks}

        yaml_result = yaml.dump(aggregated_tasks)

        print(yaml_result)

        with open(f"{self.hw_name}.yaml", "w") as f:
            f.write(yaml_result)


hw = HW([Task1(), Task2(), Task3()])
hw.evaluate()