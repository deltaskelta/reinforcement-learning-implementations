"""implementation of Jack's Car Rentla problem from chapter 4"""

import math
from typing import Iterator, Tuple

import numpy as np  # type: ignore
from matplotlib import animation  # type: ignore
from matplotlib import pyplot as plt  # type: ignore
from mpl_toolkits.mplot3d import Axes3D  # type: ignore

LOCATIONS = 2
RENT_LAMBDA = (3, 4)
RETURN_LAMBDA = (3, 2)
ACTIONS = [i for i in range(-5, 5 + 1)]
REWARD = 10
MAX_CARS = 20
MOVE_COST = 2
DISCOUNT = 0.9
EPSILON = 1e-8
DIMS = 21  # to cover 0-20 cars
POISSON_LIMIT = 10

np.set_printoptions(linewidth=140, precision=0)
fig = plt.figure()
hmap_axs = fig.add_subplot(2, 1, 1)
hmap_axs.set_xlabel("Policy")
value_axs = fig.add_subplot(2, 1, 2, projection="3d")
value_axs.set_xlabel("State Values")

# 2d xy arrays for the surface plot
x = np.arange(DIMS)
y = np.arange(DIMS)
xs, ys = np.meshgrid(x, y)


def poisson(lambd: int, k: int) -> float:
    """return the probability of k with parameter lambda in poisson distribution"""
    return float(math.exp(-lambd) * (lambd ** k / math.factorial(k)))


def valid_action(state: Tuple[int, int], action: int) -> bool:
    """return if the action is valid given the current state"""
    (i, j) = state
    if action == 0:
        return True
    if action < 0 and (i - action > 20 or j + action < 0):
        return False
    if action > 0 and (i - action < 0 or j + action > 20):
        return False
    return True


def apply_action(state: Tuple[int, int], action: int) -> Tuple[int, int]:
    """apply an action to a state, truncating any cars above 20 (does not check validity)"""
    return (
        min(MAX_CARS, int(state[0] - action)),
        min(MAX_CARS, int(state[1] + action)),
    )


def return_cars(state: Tuple[int, int]) -> Tuple[int, ...]:
    """make yesterdays expected returns available, send surplus to HQ"""
    return tuple([min(MAX_CARS, v + RETURN_LAMBDA[i]) for i, v in enumerate(state)])


class Jacks:
    """implementation of example 4.2"""

    def __init__(self) -> None:
        self.cont = True
        self.policy = np.zeros((DIMS, DIMS))
        self.state_values = np.zeros((DIMS, DIMS))
        self.i_poisson = [poisson(RENT_LAMBDA[0], i) for i in range(POISSON_LIMIT)]
        self.j_poisson = [poisson(RENT_LAMBDA[1], j) for j in range(POISSON_LIMIT)]

    def bellman(self, state: Tuple[int, int], action: int) -> float:
        """calculate the expectation of the given state and action"""

        # apply the action at the start of the day becuase we ended yesterday in this state and took the action
        # TODO: why do we have to apply the action here and not at the end?
        (i_post_action, j_post_action) = apply_action(state, action)
        expected_reward = -MOVE_COST * abs(action)

        for i in range(POISSON_LIMIT):
            for j in range(POISSON_LIMIT):
                i_state, j_state = i_post_action, j_post_action

                i_can_rent, j_can_rent = min(i, i_state), min(j, j_state)
                i_state -= i_can_rent
                j_state -= j_can_rent

                reward = 10 * (j_can_rent + i_can_rent)

                p = self.i_poisson[i] * self.j_poisson[j]
                (i_state, j_state) = return_cars(
                    (i_state, j_state)
                )  # TODO: why do we have to return cars here and not at the beginning?
                expected_reward += p * (
                    reward + DISCOUNT * self.state_values[i_state, j_state]
                )
        return expected_reward

    def evaluate_policy(self) -> None:
        """
        evaluate policy, setting values for all states according to policy. The bellman equation 
        gives expected returns under the policy, so here we continuously set the state values closer
        to the expectation (under the policy) until there is almost no difference. 
        """

        while True:
            delta = 0
            for i in range(DIMS):
                for j in range(DIMS):
                    # this starting state will have new cars become available, but the state value should
                    # reflect the value of the starting state and not the state after new cars are ready
                    v = self.state_values[i, j]
                    expectation = self.bellman((i, j), self.policy[i, j])
                    self.state_values[i, j] = expectation

                    delta = max(delta, abs(v - expectation))
            if delta < EPSILON:
                return

    def can_improve_policy(self) -> bool:
        """
        go through the policy at each state and check whether different actions can improve it
        go through all possible actions in each state and get the expected returns from the bellman equation.
        update the policy to reflect the best action. If the policy was improved that means that the valuation
        of all of the states is now different and needs to be corrected in order to reflect the updated policy
        """

        can_improve = False
        for i in range(DIMS):
            for j in range(DIMS):
                old_action = self.policy[i, j]

                best_expectation = float("-inf")
                best_action = float("-inf")
                for action in ACTIONS:
                    if not valid_action((i, j), action):
                        continue

                    expectation = self.bellman((i, j), action)
                    if expectation > best_expectation:
                        best_expectation = expectation
                        best_action = action
                self.policy[i, j] = best_action
                if self.policy[i, j] != old_action:
                    can_improve = True

        return can_improve

    def plot(self) -> None:
        """plot the updates"""
        hmap_axs.imshow(self.policy, cmap="plasma", interpolation="nearest")
        value_axs.cla()
        value_axs.plot_surface(xs, ys, self.state_values)

    def iterate_4_2(self, f: int) -> None:
        """do an iteration of policy evaluations and policy improvement"""
        self.plot()
        self.evaluate_policy()
        self.cont = self.can_improve_policy()

    def loop_until_convergence(self) -> Iterator[int]:
        """loops over the policy until convergence is reached"""
        i = 0
        while self.cont:
            i += 1
            yield i


if __name__ == "__main__":
    jacks = Jacks()
    anim = animation.FuncAnimation(
        fig, jacks.iterate_4_2, frames=jacks.loop_until_convergence
    )
    anim.save("./jacks_4_2.gif", writer="imagemagick")
