import sys
from typing import Iterator, List, Tuple, Union

import matplotlib.pyplot as plt  # type: ignore
import numpy as np  # type: ignore
from matplotlib import animation  # type: ignore
from matplotlib import cm
from mpl_toolkits.mplot3d import Axes3D  # type: ignore

CARDS = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 10, 10])

fig = plt.figure()
ace_ax = fig.add_subplot(1, 2, 2, projection="3d")
ace_ax.get_proj = lambda: np.dot(Axes3D.get_proj(ace_ax), np.diag([1.2, 1.3, 0.2, 1]))

ax = fig.add_subplot(1, 2, 1, projection="3d")
ax.get_proj = lambda: np.dot(Axes3D.get_proj(ax), np.diag([1.3, 1.3, 0.2, 1]))


def hand_value(cards: np.ndarray) -> Tuple[int, bool, bool]:
    """return the hand value, blackjack boolean, usable ace boolean"""

    if 1 in cards:
        if len(cards) == 2 and 10 in cards:
            return 21, False, True  # 21 and blackjack
        if cards.sum() - 1 > 10:
            return cards.sum(), False, False

        if cards.sum() + 10 == 21:
            return 21, False, True  # non blackjack 21, usable ace

        # we have a small card, return value with ace as 11, usable bool
        return cards.sum() + 10, False, True

    return cards.sum(), False, False


class BlackJack:
    def __init__(self) -> None:
        self.ace_states = np.zeros((10, 10))
        self.ace_states_count = np.zeros((10, 10))

        self.states = np.zeros((10, 10))
        self.states_count = np.zeros((10, 10))

        self.hand: np.ndarray
        self.dealer_hand: np.ndarray

        self.i = 0  # iterations

    def policy(self) -> bool:
        """return True (hit) if less than 20"""
        v, _, _ = hand_value(self.hand)
        return v < 20

    def dealer_policy(self) -> bool:
        """return True (hit) if less than 17"""
        v, _, _ = hand_value(self.dealer_hand)
        return v < 17

    def compare_hands(self) -> int:
        """return True if player wins, False if dealer wins, None for a draw"""
        player_v, player_bj, usable_ace = hand_value(self.hand)
        dealer_v, dealer_bj, _ = hand_value(self.dealer_hand)

        if player_bj and dealer_bj:  # both bj is a draw
            return 0
        if player_v > 21:  # player busted first, dealer always wins
            return -1
        if dealer_v > 21:  # player no bust, dealer busted
            return 1
        if dealer_v > player_v:  # dealer and player < 21 but dealer higher
            return -1
        return 1  # both < 21, only option is that player is higher

    def deal_hand(self) -> np.ndarray:
        return np.array([np.random.choice(CARDS) for i in range(2)])

    def episode(self) -> None:
        """play a hand of blackjack"""
        self.hand = self.deal_hand()
        self.dealer_hand = self.deal_hand()

        # don't need to store the states of the episode because I can just iterate
        # backwards through the hand at the end
        hit = self.policy()
        while hit:
            self.hand = np.append(self.hand, np.random.choice(CARDS))
            hit = self.policy()

        hit = self.dealer_policy()
        while hit:
            self.dealer_hand = np.append(self.dealer_hand, np.random.choice(CARDS))
            hit = self.dealer_policy()

        result = self.compare_hands()

        # update state values for the proper stores
        self.hand = np.flip(self.hand)
        for i, _ in enumerate(self.hand[:-1]):
            # the hand at the previous state (starting from the end)
            hand = self.hand[i:]
            v, _, usable_ace = hand_value(hand)
            if v < 12 or v > 21:
                continue  # meaningless to track these

            # -12 to account for 12-21 player sum
            idx = (self.dealer_hand[0] - 1, v - 12)

            if usable_ace:
                self.ace_states_count[idx] += 1
                self.ace_states[idx] += (
                    1 / self.ace_states_count[idx] * (result - self.ace_states[idx])
                )
                continue

            self.states_count[idx] += 1
            self.states[idx] += 1 / self.states_count[idx] * (result - self.states[idx])

    def iterator(self) -> Iterator[int]:
        while self.i < 500000 / 5000:
            self.i += 1
            yield self.i

    def plot(self, f: int) -> None:
        """plot the two state values every 500 iterations"""
        fig.suptitle(f"Iteration: {f * 5000}", fontsize=16)
        ace_ax.cla()
        ace_ax.title.set_text("usable ace")
        ace_x, ace_y = np.arange(10), np.arange(10)
        ace_xs, ace_ys = np.meshgrid(ace_x, ace_y)
        ace_ax.plot_surface(ace_xs, ace_ys, self.ace_states, cmap=cm.coolwarm)

        ax.cla()
        ax.title.set_text("no usable ace")
        x, y = np.arange(10), np.arange(10)
        xs, ys = np.meshgrid(x, y)
        ax.plot_surface(xs, ys, self.states, cmap=cm.coolwarm)

    def run(self, f: int) -> None:
        self.plot(f)
        for i in range(5000):
            self.episode()


if __name__ == "__main__":
    bj = BlackJack()
    anim = animation.FuncAnimation(fig, bj.run, frames=bj.iterator, interval=300)
    anim.save("./bj.gif", writer="imagemagick")
