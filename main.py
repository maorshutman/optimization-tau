from typing import Callable, Tuple

import numpy as np


def student_id():
    """
    Returns a tuple of student's ID and TAU email address.
    """
    return '305280604', r'maorshut@protonmail.com'


class QuadraticFunction:
    def __init__(
        self, 
        Q: np.ndarray, 
        b: np.ndarray
    ) -> None:
        self.Q = Q
        self.b = b

    def __call__(
        self,
        x: np.ndarray
    ) -> np.ndarray:
        return 0.5 * np.matmul(x, np.matmul(self.Q, x)) + np.matmul(self.b, x)

    def grad(
        self,
        x: np.ndarray
    ) -> np.ndarray:
        return 0.5 * np.matmul(self.Q.T + self.Q, x) + self.b

    def hessian(
        self,
        x: np.ndarray
    ) -> np.ndarray:
        return 0.5 * (self.Q.T + self.Q)


class NewtonOptimizer:

    def __init__(
        self,
        objective: Callable,
        x_0: np.ndarray,
        alpha: float,
        threshold: float,
        max_iters: int
    ) -> None:
        self.objective = objective
        self.x_0 = x_0
        self.alpha = alpha
        self.threshold = threshold
        self.max_iters = max_iters

        # These will be updated during optimization loop.
        self.curr_x = self.x_0
        self.next_x = None

    def step(self) -> Tuple:
        gx = self.objective.grad(self.curr_x)
        hx = self.objective.hessian(self.curr_x)

        # Solve the Newton system Hd = -g.
        d = -np.matmul(np.linalg.inv(hx), gx)

        self.next_x = self.curr_x + self.alpha * d

        return (self.next_x, gx, hx)

    def optimize(self) -> Tuple:

        num_iters = 0
        while num_iters < self.max_iters:

            # One step if always perfromed since the stopping creterion is |x_k+1 - x_k|.
            self.step()

            dx = self.next_x - self.curr_x
            norm_dx = np.sqrt(dx.dot(dx))

            # After `norm_dx` is computed, `curr_x is` not needed anymore.
            self.curr_x = self.next_x
            self.next_x = None

            num_iters += 1

            if norm_dx < self.threshold:
                break

            print(f"{num_iters}: ", self.curr_x, self.objective(self.curr_x))

        fmin = self.objective(self.curr_x)

        return (fmin, self.curr_x, num_iters)


class BFGSOptimizer:

    def __init__(
        self, 
        objective: Callable,
        x_0: np.ndarray,
        B_0: np.ndarray,
        alpha_0: float,
        beta: float,
        sigma: float,
        threshold: float,
        max_iters: int
    ) -> None:
        pass

    def update_dir(self) -> np.ndarray:
        pass

    def update_step_size(self) -> np.ndarray:
        pass

    def update_x(self) -> np.ndarray:
        pass

    def update_inv_hessian(
        self,
        prev_x: np.ndarray
    ) -> np.ndarray:
        pass

    def step(self) -> Tuple:
        pass

    def optimize(self) -> Tuple:
        pass


class TotalVariationObjective:

    def __init__(
        self,
        src_img: np.ndarray,
        mu: float,
        eps: float
    ) -> None:
        pass

    def __call__(
        self,
        img: np.ndarray
    ) -> np.ndarray:
        pass

    def grad(
        self,
        img: np.ndarray
    ) -> np.ndarray:
        pass


def denoise_img(
    noisy_img: np.ndarray,
    B_0: np.ndarray,
    alpha_0: float,
    beta: float,
    sigma: float,
    threshold: float,
    max_iters: int,
    mu: float,
    eps: float
) -> Tuple:
    pass


def main():

    # Q = np.eye(2)
    Q = np.array([[1., 2.],
                  [2., 1.]])

    b = np.array([2, 3])
    # b = np.array([0, 0])

    x_0 = np.array([0., 0.])

    qf = QuadraticFunction(Q, b)

    newton_opt = NewtonOptimizer(qf, x_0=x_0, threshold=1e-4 , alpha=0.1, max_iters=1000)

    print(newton_opt.optimize())

    return


if __name__ == "__main__":
    main()
