"""implementations of grid_world from chapter 3"""

from typing import Tuple

import numpy as np
import matplotlib.pyplot as plt

DIMS = 5
ACTIONS = [(0, -1), (-1, 0), (1, 0), (0, 1)]  # action tuples
PROB = 0.25
DISCOUNT = 0.9
EPSILON = 1e-1

# special position tuples
A = (0, 1)
B = (0, 3)
A_PRIME = (4, 1)
B_PRIME = (2, 3)


def plot(grid: [np.ndarray], diff: int):
    """plot the current grid and the differences between iterations"""
    text = []
    for v in grid:
        text.append(['{:1.1f}'.format(n) for n in v])

    _, axs = plt.subplots(2, 1)
    axs[0].axis('off')

    labels = [i + 1 for i in range(DIMS)]
    axs[0].table(cellText=text, colLabels=labels, rowLabels=labels, loc='center')
    axs[1].plot(diff)

    plt.show()


def step(state: Tuple[int, int], action: Tuple[int, int]) -> (Tuple[int, int], int):
    """takes the current state and action and returns next state and reward"""

    if state == A:
        return A_PRIME, 10
    if state == B:
        return B_PRIME, 5

    new_state = [v + action[i] for i, v in enumerate(state)]

    x, y = new_state
    if x < 0 or y < 0 or x > DIMS - 1 or y > DIMS - 1:
        return state, -1

    return new_state, 0


def fig_3_2():
    """implementation of the state value function in figure 3.2 of the book"""

    grid = np.zeros((DIMS, DIMS))
    diffs = list()
    while True:

        new_grid = np.zeros(grid.shape)  # all updates happen simultaneously
        for i in range(DIMS):
            for j in range(DIMS):
                for action in ACTIONS:
                    (next_i, next_j), reward = step((i, j), action)
                    new_grid[i, j] += 0.25 * (reward + DISCOUNT * grid[next_i, next_j])

        diffs.append(np.sum(np.abs(grid - new_grid)))

        if diffs[-1] < EPSILON:
            plot(new_grid, diffs)
            break

        grid = new_grid


if __name__ == "__main__":
    fig_3_2()
