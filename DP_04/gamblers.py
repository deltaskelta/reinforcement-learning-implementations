"""implementation of the Gamblers Problem using value iteration in Chapter 4"""

from typing import Iterator
from operator import add, sub
from math import isclose
import numpy as np  # type: ignore
from matplotlib import animation, pyplot as plt  # type: ignore

GOAL = 101
DIMS = 101
DISCOUNT = 1
REWARD = 1  # for reaching goal
PROB_HEADS = 0.4
OPS = [add, sub]
PROBS = [PROB_HEADS, 1 - PROB_HEADS]
EPSILON = 1e-4

np.set_printoptions(linewidth=140, precision=4)
fig, [value_axs, policy_axs] = plt.subplots(nrows=2, ncols=1)


class Gamblers():
    """implementation of example 4.2"""
    def __init__(self) -> None:
        self.cont = True
        self.policy = np.zeros(DIMS)
        self.state_values = np.zeros(DIMS)

    def bellman(self, state: int, action: int) -> float:
        """calculate the expectation of the given state and action"""
        if state == 100:
            return 1  # we win at state 100
        if state == 0 or action == 0:
            return 0  # we can't win anything with 0 money or 0 wager

        expected_reward = 0.0
        for i, op in enumerate(OPS):
            s_prime = op(state, action)
            if s_prime >= 100:
                expected_reward += PROBS[i] * REWARD  # winning state, probability of outcome * reward
                continue

            # 0 just to be explicit about current reward being 0 and DISCOUNT = 1 (no discount)
            expected_reward += PROBS[i] * (0 + DISCOUNT * self.state_values[s_prime])

        return expected_reward

    def value_iteration(self) -> bool:
        """evaluate policy, setting values for all states according to policy"""
        delta = 0
        for state in range(DIMS):
            v = self.state_values[state]
            max_return, max_index = 0.0, 0  # all returns should [0,1)
            for action in range(state + 1):
                r = self.bellman(state, action)
                if isclose(r, max_return, rel_tol=1e-8) and action > max_index:
                    continue  # there are many cases where the larger wager very close to equal to the smaller wager.
                if r > max_return:
                    max_return = r
                    max_index = action

            self.policy[state] = max_index
            self.state_values[state] = max_return
            delta = max(delta, abs(v - max_return))
        if delta < EPSILON:
            return False
        return True

    def plot(self, f: int) -> None:
        """plot the updates. f is the animate index supplied by FuncAnimation"""
        value_axs.cla()
        value_axs.set_title(f'State Values (iteration: {f})')
        value_axs.plot(self.state_values)
        policy_axs.cla()
        policy_axs.set_title('Policy/Capital')
        policy_axs.plot(self.policy)

    def iterate_4_3(self, f: int) -> None:
        """do an iteration of policy evaluations and policy improvement"""
        self.plot(f)
        self.cont = self.value_iteration()

    def loop_until_convergence(self) -> Iterator[int]:
        """loops over the policy until convergence is reached"""
        i = 0
        while self.cont:
            i += 1
            yield i


if __name__ == "__main__":
    gambler = Gamblers()
    anim = animation.FuncAnimation(fig, gambler.iterate_4_3, frames=gambler.loop_until_convergence, interval=500)
    anim.save('./gamblers_4_3.gif', writer='imagemagick')
