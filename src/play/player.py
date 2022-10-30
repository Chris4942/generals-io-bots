import torch
import play.tile as tile
import threading
import random
import time


DIRECTIONS = ["UP", "DOWN", "LEFT", "RIGHT"]
UP = 3
DOWN = 2
LEFT = 0
RIGHT = 1
LEARN_FREQUENCY = 5
MEMORY_LENGTH = 400
GAME_LENGTH = 2000
WIN_REWARD = 100
LOSS_PENALTY = -0.001
INVALID_TURN_PENALTY = -1
NO_PROGRESS_PENALTY = -0.1
TERRITORY_REWARD = 5
CONQUER_REWARD = 10

torch.set_printoptions(threshold=10000)

sigmoid = torch.nn.Sigmoid()

class Player:
    def __init__(
        self,
        general,
        setup_client,
        id,
        name,
        player_num,
        network_builder,
        epsilon,
        epsilon_decay,
        queue,
        train=True
    ):
        self.general = general # neural network that makes the decisions
        self.setup_client = setup_client
        self.id = id
        self.name = name
        self.player_num = player_num
        self.memory = []
        self.network_builder = network_builder
        self.initial_epsilon = epsilon
        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        self.last_action = None
        self.num_invalid_moves_in_game = 0
        self.turns = 0
        self.queue = queue
        self.train = train
        self.total_reward = 0

    def start_game(self):
        # print(f'player {self.player_num} is ready!')
        self.thread = threading.Thread(target = self.continuous_play)
        self.thread.start()
    
    def continuous_play(self):
        while True:
            # try:
                # self.epsilon = self.initial_epsilon
                self.memory = []
                self.turns = 0
                self.num_invalid_moves_in_game = 0
                time.sleep(0.5)
                self.client = self.setup_client(self.id, self.name)
                time.sleep(0.5)
                self.client.join_1v1()
                time.sleep(1)
                self.respond_to_changes(self.client.get_updates())
                self.client.leave()
            # except Exception as e:
            #     print(f'player{self.player_num} caught an exception. restarting continuous play.')
            #     raise e

    def height(self):
        return self.client._map.rows
    
    def width(self):
        return self.client._map.cols
    

    def respond_to_changes(self, updates_generator):
        print(f'player {self.player_num} ready to recieve updates!')
        for iteration, update in enumerate(updates_generator):
            if type(update) is str:
                self.handle_endgame(update)
                break
            y1, x1, direction, grid_tensor, action_tensor, vectorized_move = self.get_action(self.client._map.grid)

            x2, y2 = self.get_x2_y2(x1, y1, direction)
            # print(f'player_num {self.player_num} attacking from {(x1, y1)} to {(x2, y2)}')
            # print(f'player_num {self.player_num} mountain layer {grid_tensor[0]}')
            self.client.attack(y1, x1, y2, x2)
            self.remember(grid_tensor, action_tensor)
            if update.turn % LEARN_FREQUENCY == 0:
                self.learn()
            self.last_action = vectorized_move
            if iteration > GAME_LENGTH:
                self.learn()
                print(f'player_num {self.player_num} forcing game to end. num invalid moves in game = {self.num_invalid_moves_in_game}')
                break 
            # print(f'handled_turn {self.turns}', flush=True)
            self.turns += 1
        print(f'player_num {self.player_num} exited the loop for some reason')

    def get_action(self, grid):
        def decay_epsilon():
            self.epsilon *= self.epsilon_decay
        grid_tensor = self.convert_map_grid_to_tensor(grid)
        action_tensor = self.general(grid_tensor)
        if random.random() > self.epsilon:
            y1, x1, direction = self.output_tensor_to_action(action_tensor)
        else:
            owned_territory = self.find_all_owned_territory_coords(grid)
            if len(owned_territory) == 0:
                print(f'player_num: {self.player_num}: owned_territory_length = 0! {owned_territory}')
                owned_territory = [(0, 0)]
            y1, x1 = random.choice(owned_territory)
            direction = random.randint(0, 3)
        decay_epsilon()
        return y1, x1, direction, grid_tensor, action_tensor, self.grid_location_to_vectorized_location(x1, y1, direction)

    def convert_map_grid_to_tensor(self, grid):
        grid_tensor = torch.zeros([1, 5, self.height(), self.width()])
        for i, row in enumerate(grid):
            for j, t in enumerate(row):
                if t.isSelf():
                    grid_tensor[0][1][i][j] = t.army
                elif t.army > 0:
                    grid_tensor[0][2][i][j] = t.army
                elif t.tile == tile.TILE_MOUNTAIN:
                    grid_tensor[0][0][i][j] = 1
                if t.isGeneral:
                    grid_tensor[0][3][i][j] = 1
                if t.isCity:
                    grid_tensor[0][4][i][j] = 1

        return grid_tensor

    def output_tensor_to_action(self, action_tensor):
        action_distribution = action_tensor.flatten()
        # index = torch.argmax(action_distribution)
        index = torch.multinomial(torch.exp(action_distribution/0.1), 1)[0]
        self.last_action_distribution = action_distribution
        y, x, move = self.vectorized_location_to_grid_location(index)
        return y.item(), x.item(), move
    
    def vectorized_location_to_grid_location(self, index):
        height = self.height()
        width = self.width()
        move = index // (height * width)
        single_mat = index % (height * width)
        y = single_mat // width
        x = single_mat % width
        return y, x, move

    def grid_location_to_vectorized_location(self, x, y, move):
        height = self.height()
        width = self.width()
        index = move * (height * width) + y * width + x
        if index == height * width * 4:
            raise Exception(f'index is to high: {index} ({height}, {width})')
        return index

    def find_all_owned_territory_coords(self, grid):

        owned_territory = []

        for r in range(self.client._map.rows):
            for c in range(self.client._map.cols):
                if grid[r][c].isSelf() and grid[r][c].army > 1:
                    owned_territory.append((r, c))

        return owned_territory

    def get_x2_y2(self, x1, y1, direction):
        x2 = x1
        y2 = y1
        if direction == UP:
            y2 -= 1
        elif direction == DOWN:
            y2 += 1
        elif direction == RIGHT:
            x2 += 1
        elif direction == LEFT:
            x2 -= 1
        else:
            raise Exception(f'player_num: {self.player_num}: unknown direction of {direction}')
        return x2, y2

    

####### learning #######

    def remember(self, grid, action_dist):
        if len(self.memory) == 0:
            self.prev_grid = torch.zeros(grid.shape)
        reward = (torch.count_nonzero(grid[0][1]) - torch.count_nonzero(self.prev_grid[0][1])) * TERRITORY_REWARD
        # self.memory.append((self.prev_grid, self.last_action, action_dist, grid, reward))
        if self.last_action is not None:
            y1, x1, d = self.vectorized_location_to_grid_location(self.last_action)
            x2, y2 = self.get_x2_y2(x1, y1, d)
            if self.invalid_move(x2, y2, self.last_action, self.prev_grid):
                reward = INVALID_TURN_PENALTY
                self.num_invalid_moves_in_game += 1
            elif self.prev_grid[0][2][y2][x2] > 0 and grid[0][1][y2][x2] > 0:
                reward = CONQUER_REWARD
            if reward == 0:
                reward = NO_PROGRESS_PENALTY
                # print(f'player_num {self.player_num} no progress penalty applied')
            self.memory.append((
                self.prev_grid,
                self.last_action if self.last_action else 0,
                grid, 
                reward,
            0))
        # print(f'turn: {self.turns}. reward: {reward}, last_action = {self.vectorized_location_to_grid_location(self.last_action if self.last_action else 0)}')
        self.total_reward = self.total_reward + reward
        self.prev_grid = grid
    
    def invalid_move(self, x2, y2, move, grid):
        move_location = move % (self.client._map.cols * self.client._map.rows)
        raveled_index = grid[0][1].ravel()[move_location]
        if raveled_index <= 1:
            return True
        if x2 < 0 or x2 >= self.client._map.cols or y2 < 0 or y2 >= self.client._map.rows:
            return True
        if grid[0][0][y2][x2] == 1:
            return True
        return False

    
    def handle_endgame(self, update):
        magnitude = self.client._map.rows * self.client._map.cols * 4
        reward = magnitude * WIN_REWARD if update == "game_won" else magnitude * LOSS_PENALTY
        self.memory.append((self.memory[-2][2], self.last_action, self.memory[-1][2], reward, 1))
        self.learn()
    
    def learn(self):
        if self.train:
            if not self.queue.full():
                print(f'player{self.player_num} is learning. epsilon: {self.epsilon} invalid/total ={self.num_invalid_moves_in_game}/{self.turns}. Average reward = {self.total_reward/self.turns}', flush=True)
                state, action, next_state, reward, done  = zip(*self.memory)
                self.queue.put((state, action, next_state, reward, done, self.height(), self.width()))
                self.general = self.network_builder()

