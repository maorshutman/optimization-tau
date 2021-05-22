import numpy as np

from main import NewtonOptimizer
from main import QuadraticFunction
from main import BFGSOptimizer


def check_solution(x1, x2, epsilon=1e-6):
    d = x1 - x2
    norm_d = np.sqrt(d.dot(d))
    assert norm_d < epsilon


def is_pos_def(x):
    return np.all(np.linalg.eigvals(x) > 0)


def newton_opt_test_0():
    Q = np.array([[1., 2.],
                  [2., 1.]])

    b = np.array([2, 3])
    x_0 = np.array([0., 0.])

    qf = QuadraticFunction(Q, b)
    newton_opt = NewtonOptimizer(
        qf, x_0=x_0, threshold=1e-10, alpha=0.1, max_iters=1000)

    fmin, minimizer, num_iters = newton_opt.optimize()

    x_opt_expected = np.array([-1.33333333, -0.33333333])
    check_solution(x_opt_expected, minimizer)

    print(newton_opt_test_0.__name__, (fmin, minimizer, num_iters))


def newton_opt_test_1():
    Q = np.eye(2)
    b = np.array([2, 3])

    x_0 = np.array([0., 0.])

    qf = QuadraticFunction(Q, b)

    newton_opt = NewtonOptimizer(
        qf, x_0=x_0, threshold=1e-10, alpha=0.1, max_iters=1000)

    fmin, minimizer, num_iters = newton_opt.optimize()

    x_opt_expected = np.array([-2, -3])
    check_solution(x_opt_expected, minimizer)

    print(newton_opt_test_1.__name__, (fmin, minimizer, num_iters))


def newton_opt_test_2():
    """ n = 1 case """

    Q = np.eye(1)
    b = np.array([2])

    x_0 = np.array([0.])

    qf = QuadraticFunction(Q, b)

    newton_opt = NewtonOptimizer(
        qf, x_0=x_0, threshold=1e-10, alpha=0.5, max_iters=1000)

    fmin, minimizer, num_iters = newton_opt.optimize()

    x_opt_expected = - b[0] / Q[0]
    assert np.abs(x_opt_expected - minimizer) < 1e-5

    print(newton_opt_test_2.__name__, (fmin, minimizer, num_iters))


def newton_opt_test_3():
    """ Singular Hessian case. """

    Q = np.array([[1., 0.],
                  [0., 0.]])
    b = np.array([2., -3.])

    x_0 = np.array([10., -3])

    qf = QuadraticFunction(Q, b)

    newton_opt = NewtonOptimizer(
        qf, x_0=x_0, threshold=1e-10, alpha=0.5, max_iters=1000)

    fmin, minimizer, num_iters = newton_opt.optimize()

    x_opt_expected = np.array([-2, -3])
    check_solution(x_opt_expected, minimizer)

    print(newton_opt_test_3.__name__, (fmin, minimizer, num_iters))


def newton_opt_test_4():
    A = np.array([[1., 2., 1],
                  [2., 1., 3],
                  [1., 3, 7.]])
    Q = np.matmul(A, A.T)

    assert is_pos_def(Q)

    b = np.array([2, 3, -1.])
    x_0 = np.array([10., -9, 3.])

    qf = QuadraticFunction(Q, b)

    newton_opt = NewtonOptimizer(
        qf, x_0=x_0, threshold=1e-10, alpha=0.5, max_iters=1000)

    fmin, minimizer, num_iters = newton_opt.optimize()

    print(newton_opt_test_4.__name__, (fmin, minimizer, num_iters))


def bfgs_opt_test_0():
    Q = np.array([[1., 2.],
                  [2., 1.]])

    b = np.array([2, 3])
    x_0 = np.array([0., 0.])

    qf = QuadraticFunction(Q, b)

    B_0 = np.linalg.pinv(qf.hessian(x_0))

    bfgs_opt = BFGSOptimizer(
        objective=qf,
        x_0=x_0,
        B_0=B_0,
        alpha_0=1.,
        beta=0.2,
        sigma=0.3,
        threshold=1e-10,
        max_iters=1000
    )

    fmin, minimizer, num_iters = bfgs_opt.optimize()

    x_opt_expected = np.array([-1.33333333, -0.33333333])
    check_solution(x_opt_expected, minimizer)

    print(bfgs_opt_test_0.__name__, (fmin, minimizer, num_iters))


def bfgs_opt_test_1():

    A = np.array([[1., 2., 1],
                  [2., 1., 3],
                  [1., 3, 7.]])
    Q = np.matmul(A, A.T)

    assert is_pos_def(Q)

    b = np.array([2, 3, -1.])
    x_0 = np.array([10., -9, 3.])

    qf = QuadraticFunction(Q, b)

    B_0 = np.linalg.pinv(qf.hessian(x_0))

    bfgs_opt = BFGSOptimizer(
        objective=qf,
        x_0=x_0,
        B_0=B_0,
        alpha_0=1.,
        beta=0.2,
        sigma=0.3,
        threshold=1e-10,
        max_iters=1000
    )

    fmin, minimizer, num_iters = bfgs_opt.optimize()

    x_opt_expected = np.array([-0.46260388, -1.20221607,0.6565097])
    check_solution(x_opt_expected, minimizer)

    print(bfgs_opt_test_1.__name__, (fmin, minimizer, num_iters))


def bfgs_opt_test_2():
    """ Initialize Hessian with identity. """

    A = np.array([[1., 2., 1],
                  [2., 1., 3],
                  [1., 3, 7.]])
    Q = np.matmul(A, A.T)

    assert is_pos_def(Q)

    b = np.array([2, 3, -1.])
    x_0 = np.array([10., -9, 3.])

    qf = QuadraticFunction(Q, b)

    B_0 = np.eye(3)

    bfgs_opt = BFGSOptimizer(
        objective=qf,
        x_0=x_0,
        B_0=B_0,
        alpha_0=1.,
        beta=0.2,
        sigma=0.3,
        threshold=1e-10,
        max_iters=1000
    )

    fmin, minimizer, num_iters = bfgs_opt.optimize()

    x_opt_expected = np.array([-0.46260388, -1.20221607,0.6565097])
    check_solution(x_opt_expected, minimizer)

    print(bfgs_opt_test_2.__name__, (fmin, minimizer, num_iters))


def main():
    newton_opt_test_0()
    newton_opt_test_1()
    newton_opt_test_2()
    newton_opt_test_3()
    newton_opt_test_4()

    bfgs_opt_test_0()
    bfgs_opt_test_1()
    bfgs_opt_test_2()

    print()


if __name__ == "__main__":
    main()
