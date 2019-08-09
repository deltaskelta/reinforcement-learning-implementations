"""implementations of grid_world from chapter 3"""

import random
from typing import List, Tuple

import numpy as np
from matplotlib import animation, pyplot as plt

DIMS = 5
EPSILON = 1e-1
ACTIONS = [(0, -1), (-1, 0), (1, 0), (0, 1)]
DISCOUNT = 0.9
A = (0, 1)
B = (0, 3)
A_PRIME = (4, 1)
B_PRIME = (2, 3)

fig, [tbl_axs, line_axs] = plt.subplots(2, 1)


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


def plot_init() -> List[plt.Artist]:
    """make a blank plot for a new animation"""
    tbl_axs.clear()
    tbl_axs.axis('off')
    tbl_axs.table(cellText=[['0' for i in range(DIMS)] for i in range(DIMS)], loc='center')

    line_axs.clear()
    line_axs.set_xlabel('v_pi iteration differences')
    line_axs.plot([], [], 'b', animated=True)

    return tbl_axs, line_axs


class GridWorld():
    """implementation of figures 3.2 and 3.6 gridworld with equiprobable policy"""

    def __init__(self):
        self.cont = True
        self.grid = np.zeros((DIMS, DIMS))
        self.diffs = []
        self.line_x = []

    def plot(self) -> None:
        """plot the updates"""

        text = list()
        for v in self.grid:
            text.append(['{:1.1f}'.format(i) for i in v])

        tbl_axs.table(cellText=text, loc='center')
        line_axs.plot(self.line_x, self.diffs, 'b')

    def loop_until_convergence(self) -> int:
        """loops over the policy until convergence is reached"""
        i = 0
        while self.cont:
            i += 1
            yield i

    def iterate_3_2(self, f: "animate index") -> List[plt.Artist]:
        """iterate to improve the policy value with equiprobable policy"""

        new_grid = np.zeros(self.grid.shape)  # all updates happen simultaneously
        for i in range(DIMS):
            for j in range(DIMS):
                for action in ACTIONS:
                    (next_i, next_j), reward = step((i, j), action)
                    new_grid[i, j] += 0.25 * (reward + DISCOUNT * self.grid[next_i, next_j])

        self.diffs.append(np.sum(np.abs(self.grid - new_grid)))
        self.line_x.append(f)
        print(self.diffs, self.line_x)
        self.plot()

        if self.diffs[-1] < EPSILON:
            self.cont = False

        self.grid = new_grid
        return tbl_axs, line_axs

    def iterate_3_5(self, f) -> List[plt.Artist]:
        """iterate to improve the polocy value with greedy policy"""

        new_grid = np.zeros(self.grid.shape)  # all updates happen simultaneously
        for i in range(DIMS):
            for j in range(DIMS):
                best = float("-inf")
                random.shuffle(ACTIONS)  # ensure random choice of equivalued actions

                for action in ACTIONS:  # find the greedy action
                    (x, y), reward = step((i, j), action)
                    value = reward + DISCOUNT * self.grid[x, y]
                    if value > best:
                        best = value

                new_grid[i, j] += best

        self.diffs.append(np.sum(np.abs(self.grid - new_grid)))
        self.line_x.append(f)
        self.plot()

        if self.diffs[-1] < EPSILON:
            self.cont = False

        self.grid = new_grid
        return tbl_axs, line_axs


if __name__ == "__main__":

    gw = GridWorld()
    anim = animation.FuncAnimation(fig, gw.iterate_3_2, frames=gw.loop_until_convergence, init_func=plot_init)
    anim.save('grid_world_3_2.gif', writer='imagemagick')

    gw = GridWorld()
    anim = animation.FuncAnimation(fig, gw.iterate_3_5, frames=gw.loop_until_convergence, init_func=plot_init)
    anim.save('grid_world_3_5.gif', writer='imagemagick')
