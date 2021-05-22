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

        # Solve the Newton system Hd = -g. Note that we need the pseudo inverse
        # of hx in general since it can be singular.
        d = -np.matmul(np.linalg.pinv(hx), gx)

        self.next_x = self.curr_x + self.alpha * d

        return (self.next_x, gx, hx)

    def optimize(self) -> Tuple:

        num_iters = 0
        while num_iters < self.max_iters:

            # One step if always perfromed since the stopping criterion is
            # |x_k+1 - x_k|.
            self.step()

            dx = self.next_x - self.curr_x
            norm_dx = np.sqrt(dx.dot(dx))

            # After `norm_dx` is computed, `curr_x is` not needed anymore.
            self.curr_x = self.next_x
            self.next_x = None

            num_iters += 1

            if norm_dx < self.threshold:
                break

            # print(f"{num_iters}: ", self.curr_x, norm_dx, self.objective(self.curr_x))

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
        """
        Args:
            x_0       : vector, initial guess for the minimizer
            B_0       : matrix , initial guess of the inverse Hessian
            alpha_0   : scalar, initial step size for Armijo line search
            beta      : scalar, beta parameter of Armijo line search, a float
                        in range (0,1)
            sigma     : scalar, sigma parameter of Armijo line search, a float
                        in range (0,1)
            threshold : scalar, stopping criteria |𝑥_k+1 − x_k| < threshold,
                        return x_k+1
            max_iters : scalar, maximal number of iterations (stopping criteria)

        """

        self.objective = objective
        self.x_0 = x_0
        self.B_0 = B_0
        self.alpha_0 = alpha_0
        self.beta = beta
        self.sigma = sigma
        self.threshold = threshold
        self.max_iters = max_iters

        self.curr_x = self.x_0
        self.B = B_0
        self.next_x = None
        self.next_d = None
        self.alpha = None

    def update_dir(self) -> np.ndarray:
        """ Computes step direction. """
        gx = self.objective.grad(self.curr_x)
        self.next_d = -np.matmul(self.B, gx)
        return self.next_d

    def update_step_size(self) -> np.ndarray:
        """ Armijo rule. """

        h = lambda a : self.objective(self.curr_x + a * self.next_d) - self.objective(self.curr_x)
        assert h(0) == 0.

        # g = h'(0)
        # TODO: What if g >= 0 ? set g = 0 ?
        g = self.objective.grad(self.curr_x).dot(self.next_d)

        alpha = self.alpha_0

        if h(alpha) > self.sigma * g * alpha:
            # Decrease alpha.
            while True:
                alpha = alpha * self.beta
                if h(alpha) <= self.sigma * g * alpha:
                    self.alpha = alpha
                    break
        else:
            # Increase alpha.
            while True:
                prev_alpha = alpha
                alpha = alpha / self.beta
                if h(alpha) > self.sigma * g * alpha:
                    self.alpha = prev_alpha
                    break

        return self.alpha

    def update_x(self) -> np.ndarray:
        self.next_x = self.curr_x + self.alpha * self.next_d

    def update_inv_hessian(
        self,
        prev_x: np.ndarray
    ) -> np.ndarray:
        """ Rank 2 update. """

        p_k = self.alpha * self.next_d
        q_k = self.objective.grad(self.next_x) - self.objective.grad(prev_x)

        S_k = np.matmul(self.B, q_k)
        tau_k = np.matmul(S_k.T, q_k)
        mu_k = p_k.dot(q_k)

        if (mu_k < 1e-10) or (tau_k < 1e-10):
            return self.B

        v_k = p_k / mu_k - S_k / tau_k

        # print(mu_k, tau_k)
        self.B = self.B + np.outer(p_k, p_k) / mu_k - np.outer(S_k, S_k) / tau_k + tau_k * np.outer(v_k, v_k)

        return self.B

    def step(self) -> Tuple:
        self.update_dir()
        self.update_step_size()
        self.update_x()
        self.update_inv_hessian(prev_x=self.curr_x)

        return (self.next_x, self.next_d, self.alpha, self.B)

    def optimize(self) -> Tuple:
        num_iters = 0
        while num_iters < self.max_iters:

            # One step if always performed since the stopping criterion is
            # |x_k+1 - x_k|.
            self.step()

            dx = self.next_x - self.curr_x
            norm_dx = np.sqrt(dx.dot(dx))

            # After `norm_dx` is computed, `curr_x is` not needed anymore.
            self.curr_x = self.next_x
            self.next_x = None

            num_iters += 1

            if norm_dx < self.threshold:
                break

            # print(f"{num_iters}: ", self.curr_x, norm_dx, self.objective(self.curr_x))

        fmin = self.objective(self.curr_x)

        return (fmin, self.curr_x, num_iters)


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
