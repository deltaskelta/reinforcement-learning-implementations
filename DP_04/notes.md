# Jack's Car Rental

## The Bellman Algorithm

On my first attempt I tried to use the lambda values for the poisson distribution directly as the expected values of
rentals. This was shortsighted because if the inventory at a location is limited, then the expectation is lower.

For example, if the inventory is two and the expected amount of requests is three, then the first two probabilities in
the poisson distribution are 10% for 1 request and 20% for two requests, then the expected value would be

```text
1 * .10 + 2 * .9 = 1.9
```

this is because no matter how many requests come in, we only have the possibility of renting out two cars, since we only
have two in our inventory.

Add to this the fact that there are two locations with different densities for requests, and it becomes clear that we
need to take the expectation of the joint probabilities in both locations.

## The Application of the Action and Return of Cars

I also had trouble with the placement of the returning of cars and the application of the action within the Bellman
equation. My instinct was to put the return of cars at the beginning of the equation and apply the action at the "end of
the day" when we have to make a decision about tomorrow.

I am still struggling to understand why it was necessary to invert them. My feeling is that the other way would converge
on the same solution although it complicates the algorithm.

# Gambler's Problem

## Policy Iteration

There was an interesting gotcha here in the policy iteration when choosing the action with the maximum expected return.
An action that bet the whole wager at some point sometimes had an equal (or very close to equal) return to another
action with a lesser wager. This made the final graph of the policy look different than that of the book even though the
expectation would be the same. The way I got around it was to pick a tolerance for return equality and only select the
wager with the lowest value for that return.

## The Coin Probability

If the coin probability of heads is made lower, the state values start to look more exponential as they get closer to
one and the policy starts to show differences as it approaches lower amounts of capital

If the coin probability of heads goes somewhere above 0.5, the state values start to look asymptotic and the policy
starts to look more random because it has a higher chance of winning from each state. If this were the case, I think
there would have to be a negative reward for each wager in order to arrive at the best policy
