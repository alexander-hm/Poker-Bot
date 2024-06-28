# utility imports
import random
import collections
import torch
import numpy as np
from typing import Iterable

# useful pypoker modules
from pypokerengine.players import BasePokerPlayer
from pypokerengine.utils.card_utils import estimate_hole_card_win_rate, gen_cards
from PokerBotSimple import PokerBotSimple


# our netowrk
from deepQnetwork import DQNetwork

# Program settings
global DEBUG
DEBUG = False

# Model info
MEMORY_SIZE = 10000
BATCH_SIZE = 32

#Training Info
NUM_EPISODES = 1000
TARGET_LAG_FACTOR = 7500
INITIAL_STACK = 2500

# Deep RL constrants
GAMMA = 0.85 


# Greedy policy
# Probability of choosing any action at random (vs. action with highest Q value)
EPSILON = 0.1

# Helper methods
def flatten(x):
    if isinstance(x, Iterable):
        return [a for i in x for a in flatten(i)]
    else:
        return [x]      


class PokerBot2(BasePokerPlayer):

    def __init__(self, saved_model = None, training_mode_on = True):
        
        self.model_input_size = 23
        self.model_output_size = 21

        # Initialize model depending on if using saved model
        if saved_model != None:
            self.model = DQNetwork(self.model_input_size, self.model_output_size, GAMMA, saved_model=saved_model)       
        else:
             self.model = DQNetwork(self.model_input_size, self.model_output_size, GAMMA)

        
        self.training_mode_on = training_mode_on

        # set initial model states
        self.memory = collections.deque(maxlen = MEMORY_SIZE)
        self.batch_size = BATCH_SIZE
        self.first_move = True
        self.last_action = [0] * self.model_output_size
        self.last_state = [0] * self.model_input_size
        self.call_count = [0] * 5
        self.raise_count = [0] * 5
        self.fold_count = [0] * 5
        self.num_players = 0
        self.curr_stack = INITIAL_STACK
    
    # given state, select action based on model output, occasionally random
    def select_action(self, state):
        if random.uniform(0, 1) < EPSILON:
            return random.randint(0, self.model_output_size - 1)
        else:
            with torch.no_grad():
                state = torch.tensor(state, dtype=torch.float32).view(1, -1)
                q_values = self.model.predict(state)
                return torch.argmax(q_values).item()
        
    # add an experience to memory
    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    # train model on one experience
    def train_short_mem(self, state, action, reward, next_state, done):
        self.model.train_step(state, action, reward, next_state, done)

    # train model on randomly sampled batch of experiences
    def train_long_mem(self):

        sample = None
        if len(self.memory) > BATCH_SIZE:
            sample = random.sample(self.memory, BATCH_SIZE)
        else:
            sample = self.memory
            
        for state, action, reward, next_state, done in sample:
            self.model.train_step(state, action, reward, next_state, done)
       

    # Recieves: possible actions, current card, and current state of game
    # Outputs: Action
    def declare_action(self, valid_actions, hole_card, round_state):

        # Prepare feature vector based on the game state (length = 24)
        feature_vector = self._extract_features(hole_card, valid_actions, round_state)

        # If you made a last move in this round, train model on outcome of that move
        if not self.first_move and self.training_mode_on:
            self.train_short_mem(self.last_state, self.last_action, 0, feature_vector, False)
            self.remember(self.last_state, self.last_action, 0, feature_vector, False)

        
        self.first_move = False

 
        # Use the model to predict the action
        # action will be number 0-6
        ## 0 -> fold
        ## 1 -> call
        ## 2 -> 1/4 pot
        ## 3 -> 1/2 pot
        ## 4 -> 3/4 pot
        ## 5 -> pot
        ## 6 -> max raise / allin 
        action_num = self.select_action(feature_vector)

        current_pot = round_state['pot']['main']['amount']
        ## Making action map include a more balanced slate of actions TO TEST
        action_map = {
            0: 'fold', 1: 'call', 2: 'raise', 3: 'raise', 4: 'raise', 
            5: 'raise', 6: 'raise', 7: 'fold', 8: 'fold', 9: 'fold', 
            10: 'fold', 11: 'fold', 12: 'fold', 13: 'fold', 14: 'fold', 
            15: 'call', 16: 'call', 17: 'call', 18: 'call', 19: 'call', 20: 'call'
        }
        action = action_map.get(action_num, 'fold')
        amount = valid_actions[1]['amount']

        if action_num == 2:
            amount += current_pot // 4  # 1/4 pot, raising system is wierd so have to add initial bet
        elif action_num == 3:
            amount += current_pot // 2  # 1/2 pot
        elif action_num == 4:
            amount += current_pot * 3 // 4 # 3/4 pot
        elif action_num == 5:
            amount += current_pot  # full pot
        elif action_num == 6:
            amount += valid_actions[2]['amount']['max'] # all-in

        # store action for future training
        self.last_action = [0] * 21
        self.last_action[action_num] = 1
        self.last_state = feature_vector
        
        # handle case when desired action is raise, but raise is impossible given stack
        
        if amount < valid_actions[2]['amount']['min']:
            amount = valid_actions[2]['amount']['min']
        elif amount > valid_actions[2]['amount']['max']:
            amount = valid_actions[2]['amount']['max']

        if amount == -1:
            action = 'call'
            amount = valid_actions[1]['amount']

        return action, amount
    

    def receive_game_start_message(self, game_info):
        pass

    
    def _extract_features(self, hole_card, valid_actions, round_state, win = None):

        
        #simulate hand against 1000 flops extracting hand strength estimate, unless round over
        hand_strength = 0
        if win != None:
            if win:
                hand_strength = 1
            else:
                hand_strength = 0
        else:
            hand_strength = estimate_hole_card_win_rate(1000, self.num_players, gen_cards(hole_card), gen_cards(round_state['community_card']))

        # getting an estimate of pot odds strength, cant use real pot odds because inital bets have 
        # call value of 0 but using an estimate where higher number means better odds
        pot_odds = round_state['pot']['main']['amount']
        if valid_actions[1]['amount'] != 0:
            pot_odds *= 1 - (valid_actions[1]['amount'] / pot_odds)

        # get stack ammout
        stack = 0
        for player in round_state['seats']:
            if player['uuid'] == self.uuid:
                stack = player['stack']


        # 6 Standard features
 
        standard_features = [
            round_state['pot']['main']['amount'],
            stack,
            self.num_players,
            sum([side_pot['amount'] for side_pot in round_state['pot']['side']]),
            round_state['small_blind_amount'],
            self._street_to_feature(round_state['street'])
        ]

        # 15 Action history features (3 {# raises, # calls, #folds} for each betting stage: preflop, flop, turn, river, showdown)
        action_history_features = self.raise_count + self.call_count + self.fold_count

        # Combine all features into a single fixed-size feature vector of length 23
        # Flatten the list of lists
        features = flatten([hand_strength] + [pot_odds] + standard_features + action_history_features)
        features = np.array(features)
        features = features.reshape(1, -1)
        return features
    
    # Reset some game state variables at start of each round
    def receive_round_start_message(self, round_count, hole_card, seats):
        self.num_players = 0
        self.num_players  = len(seats)
        self.call_count = [0] * 5
        self.raise_count = [0] * 5
        self.fold_count = [0] * 5
        self.first_move = True

    # Not neccesarily useful
    def receive_street_start_message(self, street, round_state):
        pass

    
    # helper method: street->int
    def _street_to_feature(self, street):
        # Convert street to a numerical feature
        streets = {'preflop': 0, 'flop': 1, 'turn': 2, 'river': 3, 'showdown': 4}
        return streets.get(street, 0)
    

    # keep track of number of calls, folds, raises in round
    def receive_game_update_message(self, new_action, round_state):
        street_num = self._street_to_feature(round_state['street'])
        action = new_action['action']
        if action == 'call':
            self.call_count[street_num] += 1
        elif action == 'fold':
            self.call_count[street_num] += 1
            self.num_players -= 1
        elif action == 'raise':
            self.raise_count[street_num] += 1
    
    def receive_round_result_message(self, winners, hand_info, round_state):
        # Calculate net chip gain from round
        reward = 0
        win = False

        # record if player made money
        for w in winners:
            if w['uuid'] == self.uuid:
                win = True
        
        # calculate reward value earned by player in roudn
        for player in round_state['seats']:
            if player['uuid'] == self.uuid:
                new_stack = player['stack']
                reward = 1.75 * (new_stack - self.curr_stack)
                self.curr_stack = new_stack
        
        # train on players' last move in round
        final_state = self._extract_features(None, None, round_state, win)
        if self.training_mode_on:
            #train model with reward as net chip gain
            self.train_short_mem(self.last_state, self.last_action, reward, final_state, True)        
            self.remember(self.last_state, self.last_action, reward, final_state, True) 

    # save model to file
    def save_agent(self, file_name):
        self.model.get_NN().save(file_name)

def setup_ai(): 
    saved_model_to_test = "./models/Phil_Ivey"
    return PokerBot2(saved_model=saved_model_to_test, training_mode_on=False)

