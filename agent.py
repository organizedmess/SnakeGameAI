import torch 
import random 
import numpy as np
from collections import deque
from game import SnakeGame, Direction, Point
from model import LinearQNet, QTrainer
from helper import plot

MAX_MEMORY = 100_000
BATCH_SIZE = 1000
LR = 0.001

class Agent:
    def __init__(self):
        self.n_games = 0
        self.epsilon = 0 # randomness
        self.gamma = 0.9  # discount rate
        self.memory = deque(maxlen = MAX_MEMORY)# pops from left if the max limit is exceeded
        self.model = LinearQNet(11, 256, 3)# 11 is input, 256 are hidden, 3 are the output 
        self.trainer = QTrainer(self.model, lr=LR, gamma=self.gamma)


    def get_state(self, game):
        head = game.snake[0]
        point_l = Point(head.x - 20, head.y)# block on the left of head 
        point_r = Point(head.x + 20, head.y)# block on the right of head
        point_u = Point(head.x, head.y - 20)
        point_d = Point(head.x, head.y + 20)

        dir_l = game.direction == Direction.Left
        dir_r = game.direction == Direction.Right
        dir_u = game.direction == Direction.Up
        dir_d = game.direction == Direction.Down
        # state = [danger straight, danger right, danger left, Move Direction, Food Location]
        state = [
            # Danger Straight
            (dir_r and game.is_collision(point_r)) or 
            (dir_l and game.is_collision(point_l)) or 
            (dir_u and game.is_collision(point_u)) or 
            (dir_d and game.is_collision(point_d)), 

            # Danger Right
            (dir_r and game.is_collision(point_d)) or 
            (dir_l and game.is_collision(point_u)) or 
            (dir_u and game.is_collision(point_r)) or 
            (dir_d and game.is_collision(point_l)), 
        
            # Danger Left
            (dir_r and game.is_collision(point_u)) or 
            (dir_l and game.is_collision(point_d)) or 
            (dir_u and game.is_collision(point_l)) or 
            (dir_d and game.is_collision(point_r)), 
        
            # Move Direction 
            dir_l,
            dir_r,
            dir_u,
            dir_d,

            # Food Location
            game.food.x < game.head.x,
            game.food.x > game.head.x,
            game.food.y < game.head.y,
            game.food.y > game.head.y
        
        ]
        return np.array(state, dtype = int)
        
    def remember(self, state, action, reward, next_state, done):
        # appending in the deque
        # if its full then pop the element on the left 
        self.memory.append((state, action, reward, next_state, done))

    def train_long_memory(self):
        if len(self.memory) > BATCH_SIZE:
            mini_sample = random.sample(self.memory, BATCH_SIZE) # list of tuples
        else:
            mini_sample = self.memory
        
        states, actions, rewards, next_states, dones = zip(*mini_sample)
        # This method has divided the tuples according to their columns/elements
        self.trainer.train_step(states, actions, rewards, next_states, dones)
        
    
    def train_short_memory(self, state, action, reward, next_state, done):
        # This is for only one game step
        self.trainer.train_step(state, action, reward, next_state, done)

    def get_action(self, state):
        # The variable self.epsilon is used to control the balance between exploration and exploitation
        self.epsilon = 80 - self.n_games
        final_move = [0, 0, 0] 
        # this represents the direction that snake will take
        # [1, 0, 0] would represent left
        # [0, 1, 0] would represent straight/no change
        # [0, 0, 1] would represent right

        if random.randint(0, 200) < self.epsilon: # exploration phase as elpsilon is still large meaning we are still in early stages of n_games
            random_move = random.randint(0, 2)
            final_move[random_move] = 1
        else: # exploitation meaning that the game is now 
            # meaning that agent is now using its memory to predict the next best state/move
            state0 = torch.tensor(state, dtype=torch.float) # the input parameter 'state' is converted into a tensor
            prediction = self.model(state0)# model is taking in the input and predicting a move
            best_move = torch.argmax(prediction).item()# finding the index of the highest predicted value
            # .item() is used to convert the tensor into only one item
            final_move[best_move] = 1

        return final_move
    

def train():
    plot_scores = []
    plot_mean_scores = []
    total_score = 0
    record = 0
    agent = Agent()
    game = SnakeGame()
    while True:
        # get old state 
        state_old = agent.get_state(game)

        # get move 
        final_move = agent.get_action(state_old)

        # perform move and get new state 
        reward, done, score = game.play_step(final_move)
        state_new = agent.get_state(game)

        # train short memory 
        agent.train_short_memory(state_old, final_move, reward, state_new, done)

        # remember
        agent.remember(state_old, final_move, reward, state_new, done)

        if done: 
            # train long memory, plot result 
            game.reset()
            agent.n_games += 1
            agent.train_long_memory()

            if score > record:
                record = score
                agent.model.save()
                # agent.model.save()
            
            print('Game', agent.n_games, 'Score', score, 'Record:', record)

            plot_scores.append(score)
            total_score += score
            mean_score = total_score / agent.n_games
            plot_mean_scores.append(mean_score)
            plot(plot_scores, plot_mean_scores)


if __name__ == '__main__':
    train()