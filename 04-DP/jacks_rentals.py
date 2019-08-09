"""implementation of Jack's Car Rentla problem from chapter 4"""

from typing import Tuple
import numpy as np

LOCATIONS = 2
RENT_LAMBDA = (3, 4)
RETURN_LAMBDA = (3, 2)
ACTIONS = [i for i in range(-5, 5 + 1)]
REWARD = 10
MAX_CARS = 20
MOVE_COST = 2
DISCOUNT = 0.9
EPSILON = 1e-18
DIMS = 21  # to cover 0-20 cars

np.set_printoptions(linewidth=140, precision=0)


def is_valid_move(state: Tuple[int, int], m: int):
    """check if the proposed move is valid (new states > 0 < 20)"""
    if m == 0:
        return True

    state = return_cars(state)
    state = debit_cars(state)
    (i, j) = truncate_surplus(state)

    if m < 0 and (i - m > 20 or j + m < 0):
        return False

    if m > 0 and (i - m < 0 or j + m > 20):
        return False

    return True


def debit_cars(state: Tuple[int, int]) -> Tuple[int, int]:
    """debit the expected rented cars from the inventory"""
    return tuple([max(0, v - RENT_LAMBDA[i]) for i, v in enumerate(state)])


def truncate_surplus(state: Tuple[int, int]) -> Tuple[int, int]:
    """sends the surplus of cars back to the company HQ"""
    return tuple([min(i, MAX_CARS) for i in state])


def return_cars(state) -> Tuple[int, int]:
    """make yesterdays expected returns available, send surplus to HQ"""
    return tuple([v + RETURN_LAMBDA[i] for i, v in enumerate(state)])


def step(state: Tuple[int, int], action: int) -> (int, Tuple[int, int]):
    """take one step forward in time, return reward, and next state"""

    new_state = return_cars(state)

    # new rental requests for today, rewards being earned
    reward = 0
    for i in range(LOCATIONS):
        if new_state[i] < RENT_LAMBDA[i]:
            reward += new_state[i] * REWARD
        else:
            reward += RENT_LAMBDA[i] * REWARD

    # remove expected rented cars from inventory
    new_state = debit_cars(new_state)
    new_state = truncate_surplus(new_state)

    # follow the given action and move cars around locations
    new_state = (new_state[0] - action, new_state[1] + action)
    reward -= abs(MOVE_COST * action)
    return reward, tuple([int(i) for i in new_state])


class Jacks():
    """implementation of example 4.2"""

    def __init__(self):
        self.cont = True
        self.policy = np.zeros((DIMS, DIMS))
        self.state_values = np.zeros((DIMS, DIMS))

    def evaluate_policy(self):
        """evaluate policy, setting values for all states according to policy"""

        while True:
            delta = 0
            state_values = np.zeros((DIMS, DIMS))
            for i in range(DIMS):
                for j in range(DIMS):
                    # this starting state will have new cars become available, but the state value should
                    # reflect the value of the starting state and not the state after new cars are ready
                    v = self.state_values[i, j]
                    print(f"state: {i} {j}, policy action: {self.policy[i, j]}")
                    reward, next_state = step((i, j), self.policy[i, j])

                    value = reward + DISCOUNT * self.state_values[next_state]
                    state_values[i, j] = value

                    delta = max(delta, abs(v - value))
            self.state_values = state_values
            if delta < EPSILON:
                return

    def iterate_4_2(self, f):
        """do an iteration of policy evaluations and policy improvement"""

        while self.cont:
            self.evaluate_policy()
            self.cont = self.can_improve_policy()
            print(f"policy:\n{self.state_values}")
            print(f"value of 0, 0 cars: {self.state_values[0, 0]} value of 20, 20 cars: {self.state_values[20, 20]}")
            print(f"policy:\n{self.policy}")

    def can_improve_policy(self):
        """go through the policy at each state and check whether different actions can improve it"""

        can_improve = False
        for i in range(DIMS):
            for j in range(DIMS):
                old_action = self.policy[i, j]

                best_expectation = float("-inf")
                best_action = float("-inf")
                for n in ACTIONS:
                    if not is_valid_move((i, j), n):
                        continue

                    reward, new_state = step((i, j), n)  # expected reward for change in policy
                    expectation = reward + DISCOUNT * self.state_values[new_state]
                    if expectation > best_expectation:
                        best_expectation = expectation
                        best_action = n
                self.policy[i, j] = best_action
                if self.policy[i, j] != old_action:
                    can_improve = True

        return can_improve


if __name__ == "__main__":
    instance = Jacks()
    instance.iterate_4_2(1)


def p(i, j, string):
    if i == 0:
        print(string)
