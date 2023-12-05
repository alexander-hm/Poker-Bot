import random
import collections
import torch
import numpy as np
from typing import Iterable

from pypokerengine.players import BasePokerPlayer
from pypokerengine.utils.card_utils import estimate_hole_card_win_rate, gen_cards

from deepQnetwork import DQNetwork

# Program settings
global DEBUG
DEBUG = False
global WATCH_GAME


# Load saved model or create new
global save_model_to_file
save_model_to_file = True
global load_saved_model
load_saved_model = True
model_file_name = 'Daniel_Negreanu_FINAL'

# Model info
model_input_size = 21
model_output_size = 4
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


class PokerBotSimple(BasePokerPlayer):

    def __init__(self, saved_model = None, training_mode_on = True):
        
        # Initialize model depending on if using saved model
        if saved_model != None:
            self.model = DQNetwork(model_input_size, model_output_size, True, GAMMA)       
        else:
             self.model = DQNetwork(model_input_size, model_output_size, False, GAMMA)

        self.training_mode_on = training_mode_on
        self.memory = collections.deque(maxlen = MEMORY_SIZE)
        self.batch_size = BATCH_SIZE
        self.first_move = True
        self.last_action = [0] * model_output_size
        self.last_state = [0] * model_input_size
        self.curr_stack = INITIAL_STACK
    
    def select_action(self, state):
        if random.uniform(0, 1) < EPSILON:
            return random.randint(0, model_output_size - 1)
        else:
            with torch.no_grad():
                state = torch.tensor(state, dtype=torch.float32).view(1, -1)
                q_values = self.model.predict(state)
                return torch.argmax(q_values).item()
        
    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def train_short_mem(self, state, action, reward, next_state, done):
        self.model.train_step(state, action, reward, next_state, done)

    def train_long_mem(self):
        
        sample = None
        if len(self.memory) > BATCH_SIZE:
            sample = random.sample(self.memory, BATCH_SIZE)
        else:
            sample = self.memory
            
        for state, action, reward, next_state, done in sample:
            self.model.train_step(state, action, reward, next_state, done)
       

    def declare_action(self, valid_actions, hole_card, round_state):
        # Prepare feature vector based on the game state
        feature_vector = self._extract_features(hole_card, round_state)

        if not self.first_move and self.training_mode_on:
            self.train_short_mem(self.last_state, self.last_action, 0, feature_vector, False)
            self.remember(self.last_state, self.last_action, 0, feature_vector, False)

        self.first_move = False

        if DEBUG:
            print("input size: " + str(len(feature_vector)))
            print("input shape: " + str(feature_vector.shape))
        
        
        # Use the model to predict the action
        # action will be number 1-4
        ## 1 -> fold
        ## 2 -> call
        ## 3 -> min raise
        ## 4 -> max raise
        action_num = self.select_action(feature_vector)

        action_map = {0: 'fold', 1: 'call', 2: 'raise', 3: 'raise'}

        action = action_map.get(action_num, 0)
        amount = 0

        #get call val
        if action_num == 1:
            amount = valid_actions[1]['amount']

        # get min raise val
        if action_num == 2:
            amount = valid_actions[2]['amount']['min']
        
        # get max raise val
        if action_num == 3:
            amount = valid_actions[2]['amount']['min']
        
        self.last_action = [0, 0, 0, 0]
        self.last_action[action_num] = 1
        self.last_state = feature_vector
        
        if amount == -1:
            action = 'call'
            amount = valid_actions[1]['amount']
        return action, amount
    
    def receive_game_start_message(self, game_info):
        
        self.roudstart_stack = INITIAL_STACK

    
    def _extract_features(self, hole_card, round_state, win = None):

        
        #simulate hand against 10000 flops extracting hand strength estimate, unless round over
        hand_strength = 0
        if win != None:
            if win:
                hand_strength = 1000
            else:
                hand_strength = 0
        else:
            hand_strength = estimate_hole_card_win_rate(1000, 3, gen_cards(hole_card), gen_cards(round_state['community_card']))

        # 8 Standard features
        
        standard_features = [
            round_state['round_count'],
            round_state['pot']['main']['amount'],
            sum([side_pot['amount'] for side_pot in round_state['pot']['side']]),
            round_state['dealer_btn'],
            round_state['small_blind_pos'],
            round_state['big_blind_pos'],
            round_state['small_blind_amount'],
            self._street_to_feature(round_state['street'])
        ]

        # 8 Action history features (2 {# raises, # calls} for each betting stage: preflop, flop, turn, river)
        action_history_features = self._aggregate_action_histories(round_state['action_histories'])

        # Combine all features into a single fixed-size feature vector of length 34
        # Flatten the list of lists
        features = flatten([hand_strength] + standard_features + action_history_features)
        features = np.array(features)
        features = features.reshape(1, -1)
        return features
    
    # Not neccesarily useful
    def receive_round_start_message(self, round_count, hole_card, seats):
        for seat in seats:
            if seat['uuid']==self.uuid:
                self.roudstart_stack = seat['stack']
        self.first_move = True


    # Not neccesarily useful
    def receive_street_start_message(self, street, round_state):
        pass

    def _street_to_feature(self, street):
        # Convert street to a numerical feature
        streets = {'preflop': 1, 'flop': 2, 'turn': 3, 'river': 4, 'showdown': 5}
        return streets.get(street, 0)
    


    def _aggregate_action_histories(self, action_histories):
        '''
        # Aggregate action histories into a fixed-length vector
        # Example: Count the number of raises, calls, etc.
        raise_count = sum(1 for action in action_histories.get('preflop', []) if action['action'] == 'raise')
        call_count = sum(1 for action in action_histories.get('preflop', []) if action['action'] == 'call')
        # Add more aggregated features as needed
        # Ensure the length of this vector is fixed
        return [raise_count, call_count]
        '''
        
        # Initialize counts
        raise_count = [0, 0, 0, 0]  # Preflop, Flop, Turn, River
        call_count = [0, 0, 0, 0]
        fold_count = [0, 0, 0, 0]

        # Define rounds
        rounds = ['preflop', 'flop', 'turn', 'river']

        # Count actions in each round
        for i, round in enumerate(rounds):
            for action in action_histories.get(round, []):
                if action['action'] == 'raise':
                    raise_count[i] += 1
                elif action['action'] == 'call':
                    call_count[i] += 1
                elif action['action'] == 'fold':
                    fold_count[i] += 1

        # Flatten and return
        return raise_count + call_count + fold_count

    # Can incorporate player observation in model updated with each move
    def receive_game_update_message(self, new_action, round_state):
        pass
    
    def receive_round_result_message(self, winners, hand_info, round_state):
        # Calculate net chip gain from round
        reward = 0
        win = False
        for w in winners:
            if w['uuid'] == self.uuid:
                win = True
        for player in round_state['seats']:
            if player['uuid'] == self.uuid:
                new_stack = player['stack']
                reward = 1.75 * (new_stack - self.curr_stack)
                self.curr_stack = new_stack
        
        final_state = self._extract_features(None, round_state, win)

        if self.training_mode_on:
            #train model with reward as net chip gain
            self.train_short_mem(self.last_state, self.last_action, reward, final_state, True)        
            self.remember(self.last_state, self.last_action, reward, final_state, True) 

    def save_agent(self, file_name):
        if save_model_to_file:
            self.model.get_NN().save(file_name)

    