# Building a Deep Q-Learning Poker Bot
## Setting Up a Poker Environment
For the poker bot to engage in reinforcement learning, it needs an environment or system that allows it to simulate poker games. We used the GitHub library PyPokerEngine by Ishikota. This library is designed specifically for poker AI development and allows us to simulate games of our choosing. Additionally, it provides an abstract BasePokerPlayer class which we implement to allow our poker bot to interact with the poker engine.

## Running Instructions for GUI
1. Install all the necessary packages (pip install -r requirements.txt)
2. To run the GUI: 'pypokergui serve PATH_TO_FOLDER/poker_conf.yaml --port 8000 --speed fast'
3. To edit the bots in the GUI, editSimpleBotGui


## Deep Q-Learning
Regular Q-learning is a reinforcement learning technique in which State-Action pairs are mapped to corresponding Q-Values. These values are stored in a table and updated iteratively against target values calculated using the Bellman Equation. This state-action table method works well with finite, discrete state spaces–such as in games like chess or checkers — however, falls short at capturing the complexity of games like poker, where various state parameters exist within continuous domains, eg. pot size, call size, etc. Furthermore, poker is a game of incomplete information, meaning a clear-cut optimal move does not always exist and assumptions must be made. Therefore we turn to Deep Q-Learning in this context. Deep Q-Learning maintains a target neural network as a function approximator used to predict expected Q-values. The Bellman Equation then uses the same network to give us a MSE distance of our approximated Q-values from more exact Q values, that account directly for both immediate and future reward. The reward values are simply the amount of money the bot makes/loses directly after its action, and hence only are only non-zero after its final action of the hand.

### Training
During training interactions, the network is updated in two ways.
1. Short Term Training: This update occurs most frequently. When the bot makes action A at state S, it observes some immediate reward R and some new state S’, with possible new action A’. The degree to which S’ and A’ influence the resulting Q-value is moderated by a coefficient, with a higher value maximizing the degree to which future reward is prioritized over immediate reward, and vice versa. 
2. Deep Q-learning uses a main and target neural network to map from input states to output actions and Q-Values and enable successful learning. After passing through the network, the bot chooses the action with the highest Q-value. 

#### Hyperparameters:
* Epsilon (ε): To enable learning the bot will choose a random action with a probability of ε. This is known as the Epsilon-Greedy algorithm. Our bot uses an Epsilon value of 0.1, meaning, in expectation, every 10th move will be random and we test with a range from .1-.2. Some randomness is key for the exploration of the bot into new state-action spaces, decreasing the likelihood that our network converges on a local, but not global, minimum of the cost function.
* Gamma (γ): As shown in the Bellman Equation, γ is the degree to which future rewards (payout) are considered in measures of current reward. We experiment with gamma values from .7 - .9 during training. Before using these values, we experimented with a much lower gamma of 0.2, which over time taught the bot to almost exclusively fold, as in the short-term folding is safer than placing a bet, and the network was not influenced enough by potential future reward to take on risk in the short term. 

Both of these values are subject to future change and represent an area of the network that we can tweak to further optimize performance.

The primary features of our Deep Q-Learning implementation are as follows:


* State Representation: Encode game state into a vector with key info (bets, actions, cards).
* Q-Value Approximation: The neural network estimates future rewards for actions.
* Action Selection: Choose actions maximizing expected rewards.
* Experience Replay: Store and randomly sample past experiences for stable learning.
* Reward Design: Craft rewards for cumulative chip gains, considering poker dynamics.
* Training Iterations: The bot plays multiple rounds, refining decision-making.
* Fine-Tuning Strategies: Adapt play based on learned Q-values and opponent behavior.
* Complexity Considerations: Address poker complexities like imperfect info, bluffing, and multi-agent interactions. 

## Model Architecture
In order to make decisions from the current state of the game, the bot generates an input vector that represents the game state. We tinkered heavily with exactly what features this observation would represent, but settled ultimately on the following parameters:

* Card Strength: The bot’s approximate card strength relative to random hands against the current community cards(explained in more detail below)
* Player Actions: History of actions taken by each player at the table(e.g., fold, check, bet, raise).
* Pot Size: The total amount of chips in the pot.
Bot’s: The number of chips the bot has remaining.
* Positional Information: The position of the bot at the table (e.g., early, middle, or late position).
* Game Phase: Information about the current phase of the poker game (e.g., pre-flop, flop, turn, or river).

#### Card Strength Calculation:

The model does not know what to do with a few cards represented as strings, so to avoid this, we simulate thousands of games with the given cards to calculate the odds of winning. This used an experimental Monte Carlo approach to odds calculation rather than a combinatorics approach. This way, we can pass the model a much more useful value than cards alone.
Training and Testing
To train our model, we repeatedly had it play against copies of itself for thousands of games while varying the number of players it played against and tweaking various inputs such as the gamma, epsilon, and reward functions. Each time we trained a bot, these inputs would lead to it making noticeably different moves in testing, however, due to the complex and luck-based nature of poker, it is difficult to see which is best. One way we came up with to test this was to have our model play against both random bots and “honest bots”, where the random bot chose a random move and the honest bot bet proportionately to the strength of its hand. In our best versions, our AI could beat the random bot heads up around 80% of the time and in a game against both a random and honest bot, ours would win around 60% of the time. Furthermore, when we played against our AI ourselves, we noticed that it had developed a significant amount of “strategy” and was not by any means trivial to beat. For instance, in a heads-up game, our AI played extremely aggressively and continued to try and bully the opponent out of the game. While this strategy can lead to large losses in certain hands, this can be very effective against human players who may not be confident in their hands, especially because our bot would still fold bad cards. While our initial results are promising, we doubt that it would be able to perform consistently in games against humans, especially because it has only been trained for a few thousand generations rather than the millions it needs. Furthermore, an increased epsilon value in training would allow the model to put itself into the kind of unpredictable situations in which it must thrive to beat a human player, because as it performs currently, a strong human player would likely be able to detect patterns and tendencies of the model and use this knowledge to undermine it. 

![image](https://github.com/alexander-hm/Poker-Bot/assets/97179271/cbf1fa28-f6e0-433d-a001-a7fef75a1d8c)

In the above chart, ‘Daniel Negreanu’ refers to our model. We chose this name after one of the strongest players in world poker, who is known for his almost superhuman ability to guess the cards of other players based on their actions. 

## Next Steps
There are a number of improvements that we plan to implement to the bot. 

Find ways to optimize training speed.
* There are some inefficiencies in our bot implementation. For example, the game state feature extraction parses the entire game state into an input vector for the network every time the bot needs to take an action instead of maintaining and updating one game state vector.

Continue to tweak hyperparameters and dedicate time to training.
* Training an AI is a delicate process. More time should be spent testing various hyperparameters in training and the bot should be trained more thoroughly. Currently, we have been training the bot on laptops overnight for at most a few thousand episodes. For thorough training, we intend to train the bot for several days and closer to a hundred thousand episodes on a GPU.

Tweak reward multiplier for the cost function.
* Similar to training hyperparameters, we would like to adjust the bot’s reward multiplier — the parameter to incentivize winning during training. If the multiplier is too high, it over-incentivizes winning and results in the bot shoving (going all in) every time. If the multiplier is too low, it under-incentivizes winning and results in the bot folding every time.

Test from a human perspective to determine heuristic weaknesses.
* As we play against our trained bot in a web GUI, we can analyze its performance from a human perspective. We have observed that the bot’s behavior is too closely correlated with its chip stack. It is overly aggressive when up chips and it is too weak when chips are low. 

Potentially explore search algorithms to see further down the chain of potential future moves and consider them in the context of each other.
* Employing search algorithms is a more ambitious goal that requires a larger adjustment to the bot’s architecture. Creating a form of “move tree” that allows the bot to explore potential moves via search algorithms is a foundational technique for the most advanced poker bots currently.
