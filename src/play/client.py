import ssl
import time
import json
import threading
from websocket import create_connection, WebSocketConnectionClosedException
from play.map import Map

# A lot of the code for this client was pulled from Harris Christiansen's Python generals-bot
# Can be found here: https://github.com/harrischristiansen/generals-bot/blob/e1b1fb1c51e9f249b974a89c3e3001df530f4619/base/client/generals.py

class Client():
    def __init__(self):
        self._ws = create_connection("wss://botws.generals.io/socket.io/?EIO=3&transport=websocket", sslopt={"cert_reqs": ssl.CERT_NONE})
        self._message_to_save = []
        self._seen_update = False
        self._move_id = 1
        self.player_index = -1

        t = threading.Thread(target=self._start_sending_heartbeat)
        t.daemon = True
        t.start()
    
    def set_username(self, userid, username):
        self._send(["set_username", userid, username])
        self.username = username
        self.userid = userid

    def _send(self, msg):
        try:
            self._ws.send("42" + json.dumps(msg))
        except WebSocketConnectionClosedException:
            print("caught a websocket connection closed exception. ignoring it")

    def join_game(self, custom_game_id):
        self._send(['join_private', custom_game_id, self.userid])

    def get_updates(self):
        while True:
            try:
                msg = self._ws.recv()
            except WebSocketConnectionClosedException:
                print(f'client: web socket connection closed exception found')
                break
                
            if not msg.strip():
                print(f'breaking for some reason')
                break

            if msg in {"3", "40"}:
                continue

            while msg and msg[0].isdigit():
                msg = msg[1:]

            msg = json.loads(msg)
            if not isinstance(msg, list):
                continue
            
            if msg[0] == "error_user_id":
                print("Exit: User already in game queue")
                return
            elif msg[0] == "queue_update":
                self._log_queue_update(msg[1])
            elif msg[0] == "pre_game_start":
                print("pre_game_start")
                print(msg)
            elif msg[0] == "game_start":
                self._message_to_save.append(msg)
                self._start_data = msg[1]
                self.player_index = self._start_data['playerIndex']
                # self.save_replay_id(self._start_data.replay_id)
                print(f'start data {msg}')
            elif msg[0] == "game_update":
                #self._messagesToSave.append(msg)
                yield self._make_update(msg[1])
            elif msg[0] == "game_won":
                yield "game_won"
            elif msg[0] == "game_lost":
                yield "game_lost"
            elif msg[0] == "chat_message":
                None
            elif msg[0] == "error_set_username":
                None
            elif msg[0] == "game_over":
                None
            elif msg[0] == "notify":
                None
            else:
                print("Unknown message type: {}".format(msg))

    def set_force_start(self, queue_id):
        print("========= set force start!")
        self._send(['set_force_start', queue_id, True])
    
    def _start_sending_heartbeat(self):
        while True:
            try:
                self._ws.send("2")
            except:
                pass
            time.sleep(0.1)
    
    def _make_update(self, data):
        if not self._seen_update:
            self._seen_update = True
            self._map = Map(self._start_data, data)
            print("Joined Game!")
            return self._map
        
        return self._map.update(data)

    def _log_queue_update(self, msg):
        print(msg)
    
    def attack(self, y1, x1, y2, x2, move_half=False):
        if not self._seen_update:
            raise ValueError
        
        cols = self._map.cols
        a = y1 * cols + x1
        b = y2 * cols + x2
        self._send(["attack", a, b, move_half])
        self._move_id += 1
        # print(f'attacking from {(x1, y1)} to {(x2, y2)}')
    
    def leave(self):
        self._send(['leave_game'])

    def join_1v1(self):
        self._send(['join_1v1', self.userid])
