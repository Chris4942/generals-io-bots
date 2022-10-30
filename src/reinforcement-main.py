import os
from play.client import Client
from play.player import Player
# from generals.conv4deep3sigmoid import Conv4Deep3
from generals.extradeepconvwithcities import CityGeneral
import time
import torch
import threading
from teacher.teacher import Teacher
from multiprocessing import Queue


# save_file = "src/generals/weights/simple-conv4deep2-3-Q-learning.weights"
# save_file = "src/generals/weights/simple-conv4deep2-only-up.weights"
# load_file = 'src/generals/weights/simple-conv4deep2sigmoid.weights'
save_file = 'src/generals/weights/city-general-2-A.weights'
load_file = save_file
semaphore = threading.Semaphore()

replays_file_semaphore = threading.Semaphore()
replays_file = 'replays.txt'

def network_builder(load_weights=True,device="cpu"):
    semaphore.acquire()
    general = CityGeneral(device, 2, 32)
    if load_weights: general.load_state_dict(torch.load(load_file))
    semaphore.release()
    return general

def save_network(general):
    semaphore.acquire()
    torch.save(general.state_dict(), save_file)
    semaphore.release()

if not os.path.exists(save_file):
    save_network(network_builder(False))

queue = Queue(maxsize=6)

teacher = Teacher(queue, save_network, network_builder, "mps")
teacher_thread = threading.Thread(target = teacher.start_teaching)
teacher_thread.start()


user_ids = [word.strip() for word in os.environ['USER_IDS'].split(',')]
usernames = [word.strip() for word in os.environ['USERNAMES'].split(',')]
print(usernames)
print(user_ids)
ids_and_names = zip(user_ids, usernames)
ids_and_names_iter = iter(ids_and_names)

threads = [teacher_thread]
for index, (id, name) in enumerate(ids_and_names_iter):

    general = network_builder()
    
    def setup_player(player_num):
        def setup_client(user_id, username):
            client = Client()
            client.set_username(user_id, username)
            return client
        player = Player(
            general=general,
            setup_client=setup_client,
            id=id,
            name=name,
            player_num=player_num,
            network_builder=network_builder,
            epsilon=30/100,
            epsilon_decay=0.9991,
            queue=queue,
            train=True,
        )
        return player
    
    p1 = setup_player(index)
    time.sleep(0.1)
    p1.start_game()
    time.sleep(0.1)

    # if index > 1:
    #     pass
    #     break
    # break

for thread in threads:
    thread.join()