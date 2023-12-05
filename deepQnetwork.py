import torch
import torch.nn as nn
import torch.optim as optim


# Simple Feed forward torch network to be utilized by model
class Linear_QNet(nn.Module):

    def __init__(self, input_size, output_size):
        super().__init__()
        self.fc1 = torch.nn.Linear(input_size, 64)
        self.fc2 = torch.nn.Linear(64, 64)
        self.fc3 = torch.nn.Linear(64, output_size)


    #get file output

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        return self.fc3(x)
    
    # save model to file
    def save(self, file_name='model.pth'):
        torch.save(self.state_dict(), file_name)


# Deep Q Network: handles tr

class DQNetwork:
    def __init__(self, model_input_size, model_output_size, gamma, saved_model = None):
        # model
        self.model = Linear_QNet(model_input_size, model_output_size)
        if saved_model != None:
            print('Loading model...:' + saved_model)
            model_state = torch.load(saved_model)
            self.model.load_state_dict(model_state)               
        # set optimizer and loss functions for models
        self.optimizer = optim.Adam(self.model.parameters(), lr=0.001)
        self.criterion = nn.MSELoss()
        self.gamma = gamma

    def get_NN(self):
        return self.model
    
    # get underlying network's output for state

    def predict(self, state):
        return self.model(state)
    
    # train network on state, action, and consequences

    def train_step(self, state, action, reward, next_state, done):
        state = torch.tensor(state, dtype=torch.float32)
        next_state = torch.tensor(next_state, dtype=torch.float32)
        action = torch.tensor(action, dtype=torch.float32)
        
        
        # get model's predicted Q values at the state and make copy
        q_values = self.predict(state)
        targ = q_values.clone()


        # apply Bellman equation to get target q value for this action at this state
        updated_Q = reward
            
        if not done:
            updated_Q += self.gamma * torch.max(self.predict(next_state))

        # get the index of the action take at the state
        action_idx = torch.argmax(action).item()

        #set target q value for action at state to calculated

        targ[0][action_idx] = updated_Q 

        # train with predicted Q-values vs target

        loss = self.criterion(targ, q_values)
        
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        

