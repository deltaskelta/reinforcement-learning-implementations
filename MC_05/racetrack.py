import itertools
import sys
from typing import Dict, Tuple

import numpy as np  # type: ignore
from matplotlib import pyplot as plt  # type: ignore

# velocity is number of cells moved in a direction at each time step
# actions are increments to the velocity (-1, 0, +1) for left right and forward (no backward)
# velocities are non-negative and less than 5
# both velocities cannot be negative except at the start
# episode begins with randomly selected start state
# reward is -1 for each time step until the car crosses the finish line
# if car hits the boundary, moved back to random start, episode continues

# TODO: make the velocity change at each time step 0 with probablility 0.1

# this is a mirrored image of what is in the book on page 112
# fmt: off
track_1 = np.array([
    [0, 0, 0, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0],
    [1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0],
    [1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0],
    [1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0],
    [1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0],
    [1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0],
    [1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0],
    [1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0],
    [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0],
    [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
    [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
    [0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
    [0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
    [0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
    [0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
])

track_2 = np.array([
    [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
])
# fmt: on

EPSILON = 0.1


class Race:
    def __init__(self, track: np.array) -> None:
        self.track = track

        self.actions = np.empty((0, 3), dtype=np.int)
        for v in itertools.product([-1, 0, 1], repeat=3):
            self.actions = np.concatenate((self.actions, np.array([v])))

        # count the number of 0's of the starting line (index 0, 0) of the track
        self.start_points = np.nonzero(self.track[0])[0]
        self.finish_rows = np.nonzero(self.track[:, -1])[0]

        # position on the track and velocity 0, 1 = postition 2, 3, 4 = velocity
        self.state = np.array([0, 0, 0, 0, 0])

        # every state has every track position,
        self.policy: Dict[bytearray, np.array] = {}
        self.q: Dict[bytearray, float] = {}
        self.c: Dict[bytearray, float] = {}

    def set_velocity(self, new: np.array) -> None:
        """set the velocity and account for horizontal directions cancelling themselves out"""
        self.state[2:] += new

    def valid_action(self, new: np.array) -> bool:
        """check if the new velocity is valid or not"""
        if self.state[0] == 0 and not np.all(new == np.array([0, 1, 0])):
            return False

        for v in self.state[2:] + new:
            if v < 0 or v > 5:
                return False
        return True

    def move_sprite(self) -> None:
        """use the current state to move the sprite on the track"""
        self.state[0] += self.state[3]  # increment vertical
        self.state[1] += self.state[2] - self.state[4]

    def will_go_out_of_bounds(self, state: np.array) -> bool:
        """return true if the move will go out of bounds"""
        vert_pos = self.state[0] + self.state[3]
        hor_pos = self.state[1] + self.state[2] - self.state[4]

        left_bound = 0
        right_bound = self.track.shape[1]
        bottom_bound = self.track.shape[0]

        if hor_pos < 0:  # went off track to the left
            return True
        elif vert_pos > bottom_bound - 1:  # went off bottom
            return True
        elif (
            vert_pos in self.finish_rows and hor_pos > right_bound - 1
        ):  # went off right (but winner)
            return False
        elif self.track[vert_pos, hor_pos] == 0:
            return True
        elif self.track[vert_pos, hor_pos] == 1:
            return False
        else:
            raise ValueError(
                f"something unexected happened: {self.state}, new position: {vert_pos}, {hor_pos}"
            )

    def will_finish(self, state: np.array) -> bool:
        """return true if sprite will cross the finish line"""
        vert_pos = self.state[0] + self.state[3]
        hor_pos = self.state[1] + self.state[2] - self.state[4]

        left_bound = 0
        right_bound = self.track.shape[1]
        bottom_bound = self.track.shape[0]

        if vert_pos in self.finish_rows and hor_pos >= right_bound - 1:
            return True
        return False

    def step(self) -> int:
        """move the sprite according to the current velocity return the reward"""

        if self.will_go_out_of_bounds(self.state):
            self.reset()
            return -1
        elif self.will_finish(self.state):
            self.reset()
            return 0
        else:
            self.move_sprite()  # we only really need to move it if the game is going to continue
            return -1

    def get_soft_policy_action_prob(self, state: np.array) -> float:
        """get the probabiity of taking the soft policy action in state"""
        actions = np.array([], dtype=np.int)

        maxx = float("-inf")
        total = -1.0  # to guarantee there is no problem with dividing by 0
        action: np.array
        for i, a in enumerate(self.actions):
            bites = self.state.tostring() + a.tostring()
            if bites in self.q.keys():
                m = self.q[bites]
                total += m
                if m > maxx:
                    maxx = m
                    action = a
            else:
                self.q[bites] = 0

        return 1 - (maxx / total)

    def get_soft_policy_action(self, state: np.array) -> np.array:
        """get an action from the q function and make sure that it is a soft policy action"""
        actions = np.array([], dtype=np.int)

        maxx = float("-inf")
        total = -1.0  # to guarantee there is no problem with dividing by 0
        action: np.array
        for i, a in enumerate(self.actions):
            bites = self.state.tostring() + a.tostring()
            if bites in self.q.keys():
                m = self.q[bites]
                total += m
                if m > maxx:
                    maxx = m
                    action = a
            else:
                self.q[bites] = 0

        pr_maxx = 1 - (maxx / total)
        if maxx == float("-inf") or pr_maxx == 1.0:
            return self.actions[np.random.choice(self.actions.shape[0])]

        # pr is 1 - x because the highest negative number is the best action, it will have the smallest ratio
        # so choose this action with inverse probability
        r = np.random.choice([0, 1], p=[1 - pr_maxx, pr_maxx])
        if r == 1:
            return action
        return self.actions[np.random.choice(self.actions.shape[0])]

    def episode(self) -> Tuple[np.array, np.array, np.array]:
        states: np.array = np.empty((0, 5), int)
        actions: np.array = np.empty((0, 3), int)
        rewards: np.array = np.array([])

        while True:
            states = np.concatenate((states, self.state.reshape(-1, 5)))

            # print("state in episode: ", self.state)
            action = self.get_soft_policy_action(self.state)
            while not self.valid_action(action):
                action = self.get_soft_policy_action(self.state)

            self.set_velocity(action[:])
            actions = np.concatenate((actions, action.reshape(-1, 3)))
            rewards = np.append(rewards, self.step())
            if rewards[-1] == 0:
                return states, actions, rewards

    def reset(self) -> None:
        """set the random start point in the track"""
        self.state = np.array(
            [0, np.random.choice(self.start_points, size=1).item(), 0, 0, 0]
        )

    def run(self) -> None:
        self.reset()

        y_vals = []
        plt.ion()
        plt.show(block=True)

        x = 0
        while True:
            states, actions, rewards = self.episode()
            G = 0
            W = 1.0
            for i in reversed(range(rewards.shape[0])):
                G += rewards[i]

                bites = states[i].tostring() + actions[i].tostring()
                if bites in self.c.keys():
                    self.c[bites] += W
                else:
                    self.c[bites] = W

                self.q[bites] += W / self.c[bites] * (G - self.q[bites])

                # TODO: 0 will always be higher than -inf so I need to do something about the get_soft_policy_action
                maxx = float("-inf")
                act: np.array
                for a in self.actions:
                    bites = states[i].tostring() + a.tostring()
                    if self.q[bites] > maxx:  # should be guaranteed to exist by now
                        act = a
                self.policy[states[i].tostring()] = act

                if (actions[i] == act).all():
                    self.get_soft_policy_action_prob(
                        states[i]
                    )  # TODO: makes sure that it is not 0 somehow...?
                    W *= 1 / self.get_soft_policy_action_prob(states[i])

            x += 1
            x_vals = range(x)
            y_vals.append(G)
            # plt.gca().cla() # optionally clear axes
            plt.plot(x_vals, y_vals)
            plt.draw()
            plt.pause(0.1)


if __name__ == "__main__":
    race = Race(track_1)
    race.run()
