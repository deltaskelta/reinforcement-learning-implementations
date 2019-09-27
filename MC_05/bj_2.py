"""the second iteration of blackjack from chapter 5 (without fixed policy)"""

import sys
from typing import Iterator, List, Tuple, Union

import matplotlib.pyplot as plt  # type: ignore
import numpy as np  # type: ignore
from matplotlib import animation  # type: ignore
from mpl_toolkits.mplot3d import Axes3D  # type: ignore

CARDS = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 10, 10])

fig = plt.figure()
ace_hmap_axs = fig.add_subplot(2, 2, 1)
ace_hmap_axs.title.set_text("usable ace $\pi$")
ace_ax = fig.add_subplot(2, 2, 2, projection="3d")
ace_ax.title.set_text("usable ace $v_{\pi}$")
ace_ax.get_proj = lambda: np.dot(Axes3D.get_proj(ace_ax), np.diag([1.2, 1.3, 0.2, 1]))

ax_hmap = fig.add_subplot(2, 2, 3)
ax_hmap.title.set_text("$\pi$")
ax = fig.add_subplot(2, 2, 4, projection="3d")
ax.title.set_text("$v_{\pi}$")
ax.get_proj = lambda: np.dot(Axes3D.get_proj(ax), np.diag([1.3, 1.3, 0.2, 1]))

plt.tight_layout()


def hand_value(cards: np.ndarray) -> Tuple[int, bool, bool]:
    """return the hand value, blackjack boolean, usable ace boolean"""

    if 1 in cards:
        if len(cards) == 2 and 10 in cards:
            return 21, True, False  # 21 and blackjack
        if cards.sum() - 1 > 10:
            return cards.sum(), False, False
        # we have a small card, return value with ace as 11, usable bool
        return cards.sum() + 10, False, True
    return cards.sum(), False, False


def compare_hands(p: np.array, d: np.array) -> int:
    """
    return True if player wins, False if dealer wins, None for a draw.
    the case of blackjacks is handled in the beginning of each episode
    """
    player_v, _, _ = hand_value(p)
    dealer_v, _, _ = hand_value(d)

    if player_v > 21:  # player busted first, dealer always wins
        return -1
    if dealer_v > 21:  # player no bust, dealer busted
        return 1
    if dealer_v > player_v:  # dealer and player < 21 but dealer higher
        return -1
    if dealer_v == player_v:
        return 0
    return 1  # both < 21, only option is that player is higher


def deal() -> np.ndarray:
    return np.array([np.random.choice(CARDS) for i in range(2)])


class BlackJack:
    def __init__(self) -> None:
        # start with an initial count so that an initial iteration can't bump
        # it all the way to -1, also start with all values at 1 to encourage exploration
        self.policy = np.zeros((10, 10))
        self.policy[:, :8] = 1
        self.q_count = np.ones((10, 10, 2)) * 20
        self.q = np.ones((10, 10, 2))

        self.ace_policy = np.zeros((10, 10))
        self.ace_policy[:, :8] = 1
        self.ace_q_count = np.ones((10, 10, 2)) * 20
        self.ace_q = np.ones((10, 10, 2))

        self.hand: np.ndarray
        self.dealer_hand: np.ndarray

        self.i = 0  # iterations

    def make_v_from_q(self, q: np.array) -> np.array:
        """input whichever q we want to make a v for"""
        v: np.array = np.zeros((10, 10))
        for i in range(10):
            for j in range(10):
                v[i, j] = max(q[i, j])
        return v

    def get_policy_action(self) -> int:
        """get a policy action from the q values"""
        p_val, _, ace = hand_value(self.hand)
        if p_val < 12:
            return 1
        if p_val >= 21:
            return 0

        idx = (self.dealer_hand[0] - 1, p_val - 12)
        if ace:
            return int(np.argmax(self.ace_q[idx]))
        return int(np.argmax(self.q[idx]))

    def dealer_policy(self) -> bool:
        """return True (hit) if less than 17"""
        v, _, _ = hand_value(self.dealer_hand)
        return v < 17

    def deal_hand(self) -> None:
        self.hand = deal()
        self.dealer_hand = deal()

    def episode(self) -> None:
        """play a hand of blackjack"""
        # don't need to store the states of the episode because I can just iterate
        # backwards through the hand at the end
        self.deal_hand()

        p_v, p_bj, _ = hand_value(self.hand)
        d_v, d_bj, _ = hand_value(self.dealer_hand)

        if p_bj and d_bj:
            self.update(self.hand, 0, 0)
        elif d_bj:
            self.update(self.hand, 0, -1)
        elif p_bj:
            self.update(self.hand, 0, 1)
        else:
            actions: List[int] = []
            actions.append(self.get_policy_action())
            while actions[-1]:
                v, _, _ = hand_value(self.hand)
                self.hand = np.append(self.hand, np.random.choice(CARDS))
                actions.append(self.get_policy_action())

            hit = self.dealer_policy()
            while hit:
                self.dealer_hand = np.append(self.dealer_hand, np.random.choice(CARDS))
                hit = self.dealer_policy()

            result = compare_hands(self.hand, self.dealer_hand)

            self.hand = np.flip(self.hand)
            actions = np.flip(actions)
            for i, _ in enumerate(self.hand[:-1]):
                # the hand at the previous state (starting from the end)
                hand = self.hand[i:]
                self.update(hand, actions[i], result)

    def update(self, p_hand: np.array, action: int, r: int) -> None:
        """update the policies according to the state and reward"""
        v, _, ace = hand_value(p_hand)
        if v < 12 or v >= 21:
            return

        idx = (self.dealer_hand[0] - 1, v - 12, action)
        pi_idx = idx[:-1]

        if ace:
            self.ace_q_count[idx] += 1
            self.ace_q[idx] += 1 / self.ace_q_count[idx] * (r - self.ace_q[idx])
            self.ace_policy[pi_idx] = np.argmax(self.ace_q[pi_idx])
            return

        self.q_count[idx] += 1
        self.q[idx] += 1 / self.q_count[idx] * (r - self.q[idx])
        self.policy[pi_idx] = np.argmax(self.q[pi_idx])

    def iterator(self) -> Iterator[int]:
        while self.i < 2000000 / 10000:
            self.i += 1
            yield self.i

    def plot(self, f: int) -> None:
        """plot the two state values every 500 iterations"""
        fig.suptitle(f"Iteration: {f * 10000}", fontsize=16)

        ace_hmap_axs.imshow(self.ace_policy, cmap="plasma", interpolation="nearest")

        ace_ax.cla()
        ace_ax.title.set_text("usable ace")
        ace_x, ace_y = np.arange(10), np.arange(10)
        ace_xs, ace_ys = np.meshgrid(ace_x, ace_y)
        ace_ax.plot_surface(
            ace_xs, ace_ys, self.make_v_from_q(self.ace_q), cmap="plasma"
        )

        ax_hmap.imshow(self.policy, cmap="plasma", interpolation="nearest")

        ax.cla()
        ax.title.set_text("no usable ace")
        x, y = np.arange(10), np.arange(10)
        xs, ys = np.meshgrid(x, y)
        ax.plot_surface(xs, ys, self.make_v_from_q(self.q), cmap="plasma")

    def run(self, f: int) -> None:
        self.plot(f)
        for i in range(10000):
            self.episode()


if __name__ == "__main__":
    bj = BlackJack()
    anim = animation.FuncAnimation(
        fig, bj.run, frames=bj.iterator, interval=200, save_count=200
    )
    anim.save("./bj-2.gif", writer="imagemagick")
