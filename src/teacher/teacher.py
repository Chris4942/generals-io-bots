from multiprocessing import Queue
import torch
from time import perf_counter

mse_loss = torch.nn.MSELoss()
gamma = 0.99
global_step = 0
target_update = 1000

class Teacher:
    def __init__(self, queue: Queue, publish_student, network_builder, device):
        self.student = network_builder(device=device)
        self.queue = queue
        self.publish_student = publish_student
        self.network_builder = network_builder
        self.average_time = 9999999999
        self.device = device
    
    def start_teaching(self):
        self.optim = torch.optim.Adam(self.student.parameters())
        self.target_network = self.network_builder(device=self.device)
        self.target_network.load_state_dict(self.student.state_dict())
        global_step = 0
        self.total_time = 0
        self.number_of_training_loops = 0
        while True:
            start_time = perf_counter()
            if global_step % target_update == 0:
                self.target_network.load_state_dict(self.student.state_dict())
            try:
                teaching_materials = self.prepare_batch()
                self.teach(teaching_materials)
            except Exception as e:
                print(f'caught exception in training loop. Just skipping this batch: {e}')
            elapsed_time =(perf_counter() - start_time)
            self.total_time += elapsed_time
            self.number_of_training_loops += 1
            self.average_time = (self.total_time / self.number_of_training_loops)

    def teach(self, batch):
        self.learn_dqn(batch, self.optim, self.student, self.target_network, gamma, global_step, target_update)
        self.publish_student(self.student)


    def prepare_batch(self):
        state, action, next_state, reward, done, height, width = self.queue.get()
        self.height = height
        self.width = width
        state = torch.cat(state).to(self.device)
        action = torch.tensor(action).to(self.device)
        next_state = torch.cat(next_state).to(self.device)
        reward = torch.tensor(reward).float().to(self.device)
        done = torch.tensor(done).float().to(self.device)
        return state, action, next_state, reward, done

    def learn_dqn(self, batch, optim, q_network, target_network, gamma, global_step, target_update):
        with torch.autograd.set_detect_anomaly(True):
            state, action, next_state, reward, done = batch
            optim.zero_grad()
            q_values = q_network(state)
            length = len(q_values)

            # one hot encode
            one_hot = torch.zeros([len(state), self.height*self.width*4]).to(self.device)
            unsqueezed_action = action.unsqueeze(1)
            try:
                one_hot.scatter_(1, unsqueezed_action, 1)
            except:
                print(f'error in memory! deleting memory and returning')
                self.memory = []
                return
            one_hot = one_hot.bool()

            t_q_values = target_network(next_state)[0:length]
            max_values, _ = torch.max(t_q_values, dim=1)
            target = reward + gamma * max_values*(1-done)
            q_values_one_hots = q_values[one_hot]
            if q_values_one_hots.shape != target.shape:
                print(f'Teacher Error: q_values.shape {q_values_one_hots.shape} != target.shape {target.shape}')
            loss = mse_loss(q_values_one_hots, target)
            print(f'Teacher Report: loss {loss}, queue size= {"unknown"}. Average time per iteration = {self.average_time}s. {self.number_of_training_loops}/{self.total_time}', flush=True)
            if self.total_time < 0:
                print("somehow negative?")
            loss.backward()
            optim.step()
