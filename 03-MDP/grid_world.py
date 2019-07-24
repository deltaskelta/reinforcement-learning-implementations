"""implementations of grid_world from chapter 3"""

import random
from typing import List, Tuple

import numpy as np
from matplotlib import animation, rc, pyplot as plt

rc('animation', html='html5')

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

#def plot(grid: [np.ndarray], diff: int):
#    """plot the current grid and the differences between iterations"""
#    text = []
#    for v in grid:
#        text.append(['{:1.1f}'.format(n) for n in v])
#
#    _, axs = plt.subplots(2, 1)
#    axs[0].axis('off')
#
#    labels = [i + 1 for i in range(DIMS)]
#    axs[0].table(cellText=text, colLabels=labels, rowLabels=labels, loc='center')
#    axs[1].set_xlabel('v_pi iteration differences')
#    axs[1].plot(diff)
#
#    plt.show()

fig, axs = plt.subplots(2, 1)


def plot(grid: np.ndarray, diff: List[int]) -> 'matplotlib.image.AxesImage':
    """makes an image of the current state"""

    text = list()
    for v in grid:
        text.append([['{:1.1f}'.format(i) for i in v]])

    axs[0].axis('off')
    axs[0].table(cellText=text, loc='center')
    axs[1].set_xlabel('v_pi iteration differences')
    axs[1].plot(diff)

    return plt.imshow(axs)


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


def fig_3_2() -> 'matplotlib.image.AxesImage':
    """implementation of the state value function (v_pi(s)) in figure 3.2 of the book. Same probability for each action"""

    grid = np.zeros((DIMS, DIMS))
    images = list()
    diffs = list()
    while True:

        new_grid = np.zeros(grid.shape)  # all updates happen simultaneously
        for i in range(DIMS):
            for j in range(DIMS):
                for action in ACTIONS:
                    (next_i, next_j), reward = step((i, j), action)
                    new_grid[i, j] += 0.25 * (reward + DISCOUNT * grid[next_i, next_j])

        diffs.append(np.sum(np.abs(grid - new_grid)))
        images.append(plot(new_grid, diffs))

        if diffs[-1] < EPSILON:
            print("breaking")
            print(grid, new_grid, diffs)
            break

        grid = new_grid
    return images


def fig_3_5() -> 'matplotlib.image.AxesImage':
    """implementation of the state value function (v_pi(s)) in figure 3.2 of the book. Always takes greedy action"""

    grid = np.zeros((DIMS, DIMS))
    diffs = list()
    while True:

        new_grid = np.zeros(grid.shape)  # all updates happen simultaneously
        for i in range(DIMS):
            for j in range(DIMS):
                best = float("-inf")
                random.shuffle(ACTIONS)  # ensure random choice of equivalued actions

                for action in ACTIONS:  # find the greedy action
                    (x, y), reward = step((i, j), action)
                    value = reward + DISCOUNT * grid[x, y]
                    if value > best:
                        best = value

                new_grid[i, j] += best

        diffs.append(np.sum(np.abs(grid - new_grid)))
        if diffs[-1] < EPSILON:
            break

        grid = new_grid


if __name__ == "__main__":

    imgs = fig_3_2()
    print("after return")
    anim = animation.ArtistAnimation(fig, imgs, blit=True)
    plt.show()
    anim.save('../images/grid_world_3_5.gif', writer='imagemagick', fps=15)

    #fig_3_5()
