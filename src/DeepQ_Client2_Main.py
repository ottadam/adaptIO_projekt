#encoding: utf-8
# felhasznált forrás: https://github.com/mswang12/minDQN/blob/main/minDQN.py
# Ottmár Ádám, Orbázi Tibor

# használt modulok beimportálása
import time
from Client import SocketClient
import json
import numpy as np
import math
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow import keras
from collections import deque
import random

# randomszám generálás paramétereinek beállítása
RANDOM_SEED = 5
tf.random.set_seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)

# tanítás és visszatöltés beállításai
Training_ON = False
Load_model_ON = True
Save_model_ON = False

train_episodes = 50 # játszott játékok száma
replay_memory = deque(maxlen = 50_000) # memória mérete
    
# mentés, visszaolvasás elérési útvonala
PATH = "D:/BME/Msc/1. félév\Adaptív rendszerek modellezése/Porjekt/adaptivegame/AI_mentesek_0513" 


# függvény a neurális háló feépítésére
def create_model():
    # bemeneti réteg 410 neuron, első rejtett réteg 120 neuron, másiodik rejtett réteg 60 neuron, és 9 kimenet, aktivációs függvény relu
    learning_rate = 0.01
    init = tf.keras.initializers.HeUniform()
    model = keras.Sequential()
    model.add(keras.layers.Dense(120, input_shape=(410,), activation='relu', kernel_initializer=init))
    model.add(keras.layers.Dense(60, activation='relu', kernel_initializer=init))
    model.add(keras.layers.Dense(9, activation='linear', kernel_initializer=init))
    model.compile(loss=tf.keras.losses.Huber(), optimizer=tf.keras.optimizers.Adam(lr=learning_rate), metrics=['accuracy'])
    
    return model


# függvény a háló tanítására
def train(replay_memory, model, target_model):
    #print("\n training started")
    learning_rate = 0.7 # lr a Bellman egyenlethez
    discount_factor = 0.62 # Bellman egyenlethez

    MIN_REPLAY_SIZE = 300  # csak akkor tanulunk ha van legalább 300 korábbi állapot-akció-jutalom lementve
    if len(replay_memory) < MIN_REPLAY_SIZE:
        #print('replay memory small')
        return

    batch_size = 150  # 150 random elem kiválasztása a memóriából és szétbontása, ezeken tanítunk
    mini_batch = random.sample(replay_memory, batch_size)
    current_states = np.array([transition[0] for transition in mini_batch])
    current_qs_list = model.predict(current_states)
    new_current_states = np.array([transition[3] for transition in mini_batch])
    future_qs_list = target_model.predict(new_current_states)  # a target háló kimenete
    #print(' future_qs_list shape ', future_qs_list.shape)

    X = []
    Y = []
    
    for index, (observation, action, reward, new_observation) in enumerate(mini_batch):  # végig szaladunk a 100 elemen
        #if not done:
        #print("\n index ", index)
        max_future_q = reward + discount_factor * np.max(future_qs_list[index])  # a várható legnagyobb q érték
        #print(' max_future_q ', max_future_q)
        
        current_qs = current_qs_list[index] # aktuális q értékek
        #print(' current_qs ', current_qs)
        current_qs[action] = (1 - learning_rate) * current_qs[action] + learning_rate * max_future_q  # Bellman egyenlet
        
        X.append(observation)
        Y.append(current_qs)
    #print(' X shape', len(X))
    #print(' Y shape', len(Y))
    model.fit(np.array(X), np.array(Y), batch_size=batch_size, verbose=0, shuffle=True) # tanítás a 100 mintán



# Neurális háló bot osztálya
class RemoteAI:
    
    def __init__(self, path=None):
        # Osztály saját változóinak inicializálása
        self.state = np.zeros((410)) # környezet állapota
        self.action = 0 # ágens lépése
        
        self.episode = 1 # aktuális játék száma
        self.episode_reward = 0 # játékban kapott összes jutalom
        self.reward = 0 # lépésért kapott jutalom
        self.sum_reward = 0 # összes jutalom
        self.sum_score = 0 # összesített pontszám
        
        self.oldpos = 0 # korábbi és aktuális pozíció és méret letárolására
        self.no_move_steps = 0
        self.oldposh = None
        self.oldcounter = 0
        self.size = 0
        self.oldsize = 0
         
        self.total_rewards=np.zeros(train_episodes) # háló jutalmainak és pontszámának követéséhez
        self.total_score=np.zeros(train_episodes)
        self.total_avg_reward=np.zeros(train_episodes)
        self.total_avg_score=np.zeros(train_episodes)
       
        self.steps_to_update_target_model = 1 # target háló frissítésére számláló változó
        
        if Training_ON == True:
            self.epsilon = 1  # epsilon greedy algoritmus változói
            self.max_epsilon = 1
            self.min_epsilon = 0.05 
            self.decay = 0.01
        else:
            self.epsilon = 0.05  # epsilon greedy algoritmus változói
            self.max_epsilon = 0.05 
            self.min_epsilon = 0.05 
            self.decay = 0.00
    

     # random lépést generáló függvény
    def getRandomAction(self):
         r = np.random.randint(0, 8)
         action = r
         return action
     
     # NaiveHunterAction logikájával lépést generáló függvény
    def NaiveHunterAction(self, fulljson):
        jsonData = fulljson["payload"] 
        if self.oldposh is not None:
             if tuple(self.oldposh) == tuple(jsonData["pos"]):
                 self.oldcounter += 1
             else:
                 self.oldcounter = 0
        if jsonData["active"]:
             self.oldposh = jsonData["pos"].copy()

        vals = []
        for field in jsonData["vision"]:
             if field["player"] is not None:
                 if tuple(field["relative_coord"]) == (0, 0):
                     if 0 < field["value"] <= 3:
                         vals.append(field["value"])
                     elif field["value"] == 9:
                         vals.append(-1)
                     else:
                         vals.append(0)
                 elif field["player"]["size"] * 1.1 < jsonData["size"]:
                     vals.append(field["player"]["size"])
                 else:
                     vals.append(-1)
             else:
                 if 0 < field["value"] <= 3:
                     vals.append(field["value"])
                 elif field["value"] == 9:
                     vals.append(-1)
                 else:
                     vals.append(0)

        values = np.array(vals)
         # print(values, fieldDict["vision"][np.argmax(values)]["relative_coord"], values.max())
        if np.max(values) <= 0 or self.oldcounter >= 3:
             self.nextAction = self.getRandomAction()
             self.oldcounter = 0
             
             return self.nextAction
         
        else:
             idx = np.argmax(values)
             actstring = ""
             for i in range(2):
                 if jsonData["vision"][idx]["relative_coord"][i] == 0:
                     actstring += "0"
                 elif jsonData["vision"][idx]["relative_coord"][i] > 0:
                     actstring += "+"
                 elif jsonData["vision"][idx]["relative_coord"][i] < 0:
                     actstring += "-"

       # választott lépés dekódolása
        if actstring == "-+":
             action = 0
        elif actstring == "0+":
             action = 1
        elif actstring == "++":
             action = 2
        elif actstring == "-0":
             action = 3
        elif actstring == "00":
             action = 4
        elif actstring == "+0":
             action = 5
        elif actstring == "--":
             action = 6
        elif actstring == "0-":
             action = 7
        elif actstring == "+-":
             action = 8
        else:
             action = 0
             
        return action
    
    
    # függvény a lépések utánni új állapot feldolgozására és a jutalom meghatározására
    def get_state_and_reward(self, fulljson):
        
        # használt változók inicializálása
        reward_food = 0 
        reward_hunt = 0
        reward_avoid = 0
        reward_alive = 0
        reward_moved = 0
        reward_size = 0
        reward_wall = 0  
        new_state = np.zeros(410)
        
        
        jsonData = fulljson["payload"]
        if "pos" in jsonData.keys() and "tick" in jsonData.keys() and "active" in jsonData.keys() and "size" in jsonData.keys() and "vision" in jsonData.keys():
            
            new_state[1] = (jsonData["pos"][0]/40) # játékos x pozíciója 0-1 közé normalizálva
            new_state[2] = (jsonData["pos"][1]/40) # játékos y pozíciója 0-1 közé normalizálva
            new_state[0] = (jsonData["size"]/500) # játékos mérete 0-1 közé normalizálva

            # jutalom ha megmozdult az ágens (büntetés ha nem)
            if self.oldpos == jsonData["pos"]:
                reward_moved = 0
                self.no_move_steps += 1
            else: 
                reward_moved = 1
                self.no_move_steps = 0
            
            # jutalom ha életben van az ágens, (büntetés ha nem)
            if jsonData["active"]:
                reward_alive = 1
                self.oldpos = jsonData["pos"] # utolsó pozíció frissítése
                self.size = jsonData["size"] # utolsó méret frissítése
            else: reward_alive = 0
            
            # Ha nőtt az ágens mérete akkor nagy jutalom a növekedéssel aréányosan
            if self.oldsize < self.size:
                #print(self.oldsize, self.size)
                reward_size = (self.size - self.oldsize)*3
                self.oldsize = self.size
            else:
                reward_size = 0
            
            # látómező feldolgozása
            # a látómező pixeleit one-hot encoding-al tároljuk. 
            # a vektor 1. eleme 1 ha a mező üres
            # a vektor 2. eleme 1 ha 1 értékű kaja van a mezőben 
            # a vektor 3. eleme 1 ha 2 értékű kaja van a mezőben 
            # a vektor 4. eleme 1 ha 3 értékű kaja van a mezőben 
            # a vektor 5. eleme 1 ha fal van a mezőben
            # a vektor minden más eleme 0
            # a vektorok összefűzve 81*5 = 405 elemű tömböt alkotnak
            i = 0
            for field in jsonData["vision"]:
                if field["player"] is not None:  # ha van ellenfél
                    if tuple(field["relative_coord"]) == (0, 0):  #akik nem mi vagyunk
                        if 0 < field["value"] <= 3:  # kajás mezők feldolgozása
                            t = int(field["value"])
                            new_state[5*i+3+t] = 1
                            reward_food = reward_food + field["value"]/20 # jutalom ha kaja van a látómezőben
                        elif field["value"] == 9: # falak feldolgozása
                            new_state[5*i+3+4] = 1
                            reward_wall -= (math.sqrt(81)-(math.sqrt(math.pow((field["relative_coord"][0]),2)+math.pow((field["relative_coord"][1]),2))))/20  # büntetés ha fal van a látómezőben
                        else:           # ürres mező
                            new_state[5*i+3] = 1
                        
                    else: 
                        new_state[408] = i/80   # ellenfél pozíciója 0-1 közé normálva
                        new_state[409] = field["player"]["size"]/500 # ellenfél mérete 0-1 közé normálva
                        #print('\n new_state: ', new_state)
                        #print(math.pow(field["relative_coord"][0],2), math.pow(field["relative_coord"][1],2))
                        if field["player"]["size"] * 1.1 < jsonData["size"]:
                            reward_hunt = (math.sqrt(81)-(math.sqrt(math.pow((field["relative_coord"][0]),2)+math.pow((field["relative_coord"][1]),2))))/2 # jutalom ha elkerüli a nála nagyobb ellenfelet
                            reward_avoid = 0
                        elif field["player"]["size"] > jsonData["size"]* 1.1:
                            reward_hunt = 0
                            reward_avoid = math.sqrt(math.pow((field["relative_coord"][0]),2)+math.pow((field["relative_coord"][1]),2))/2 # jutalom ha vadászik a kisebb ellenfélre
                        else:
                            reward_hunt = 0
                            reward_avoid = 0
                else: 
                    if 0 < field["value"] <= 3: # kajás mezők feldolgozása
                        t = int(field["value"])
                        new_state[5*i+3+t] = 1
                        reward_food = reward_food + field["value"]/20 # jutalom ha kaja van a látómezőben
                    elif field["value"] == 9:  # falak feldolgozása
                        new_state[5*i+3+4] = 1
                        reward_wall -= (math.sqrt(81)-(math.sqrt(math.pow((field["relative_coord"][0]),2)+math.pow((field["relative_coord"][1]),2))))/20 # büntetés ha fal van a látómezőben
                    else:
                        new_state[5*i+3] = 1 # falak feldolgozása
                        
                #○print('\n i: ', i, 'new_state: ', new_state)
                i += 1
            
            #print('\n reward_size: ',reward_size,  ' reward_food: ',reward_food,' reward_hunt: ',reward_hunt,' reward_avoid: ',reward_avoid,' reward_alive: ',reward_alive)
            reward = reward_size + reward_food + reward_hunt + reward_avoid# + reward_wall # + reward_moved # + reward_alive # összesített jutalom kiszámítása
            #print(reward_moved)
            
            if reward_alive == 0: # ha megették az ágenst akkor jutalom = 0
                reward = 0
            
            return new_state, reward
        

    # Az egyetlen kötelező elem: A játékmestertől jövő információt feldolgozó és választ elküldő függvény
    def processObservation(self, fulljson, sendData):
         # játék elindítása masterként, ha a játék indításra kész
         if fulljson["type"] == "readyToStart" and self.episode < train_episodes:
             time.sleep(1)
             sendData(json.dumps({"command": "GameControl", "name": "master",
                                  "payload": {"type": "start", "data": None}}))
             self.state = np.zeros((410)) # újrainicializálás játékok elején
             self.episode_reward = 0
             self.no_move_steps = 0
             self.episode = self.episode + 1
             print("Game is ready, starting in 5")
             
            # ha a játék elindult kiirítás
         if fulljson["type"] == "started":
             print("Startup message from server.")
             print("Ticks interval is:",fulljson["payload"]["tickLength"])
             
        # játékadatok feldolgozása
         elif fulljson["type"] == "gameData":    
            # ha a játékos életben van
             if fulljson["payload"]["active"]:
                 actstring = ''
                 
                 # epsilon greedy algoritmus, ha a random szám kisebb mint epszilon, akkor random lépés, ha nagyobb akkor a háló lép
                 e =  np.random.rand()
                 if e <= self.epsilon or self.no_move_steps >= 2:
                     f =  np.random.rand()
                     if f <= 0.5: #if self.episode >= 0:    
                         self.action = self.getRandomAction() # random lépés
                         #print('random_move', self.action)
                     else:
                         self.action = self.NaiveHunterAction(fulljson)  # néha a naivehunter_bot is lépéshez jut
                         #print('naivehunter_move', self.action)
                 else:
                    environment = self.state # környezet állapota
                    environment_reshaped = environment.reshape([1, environment.shape[0]]) 
                    predicted = model.predict(environment_reshaped).flatten() # neurális háló lépése
                    self.action = np.argmax(predicted) # akció meghatározása
                    #print('AI_move', self.action)
               
                # választott akció kódolása az elvártg módon
                 if self.action == 0:
                     actstring = "-+"
                 elif self.action == 1:
                     actstring = "0+"
                 elif self.action == 2:
                     actstring = "++"
                 elif self.action == 3:
                     actstring = "-0"
                 elif self.action == 4:
                     actstring = "00"
                 elif self.action == 5:
                     actstring = "+0"
                 elif self.action == 6:
                     actstring = "--"
                 elif self.action == 7:
                     actstring = "0-"
                 elif self.action == 8:
                     actstring = "+-"
                 else:
                     actstring = "00"
                     
                 #print('move:',actstring)
                 sendData(json.dumps({"command": "SetAction", "name": "RemotePlayer", "payload": actstring})) # választott lépés kiküldése
              
                 new_state, self.reward = self.get_state_and_reward(fulljson) # kapott jutalom és a lépés utáni új környezet letárolása
                 if math.isnan(self.reward):
                     self.reward = 0
                 
                 #print('state:', self.state)
                 #print('new_state', new_state)
                 #print('action: ', self.action, 'reward: ', self.reward)
                 
                 if Training_ON == True: # ha a tanulás be van kapcsolva
                     self.margin = np.sum(self.total_rewards)/(self.episode*300)*0.8; # határérték kiszámolása
                     
                     if len(replay_memory) < 5000:     #az első 5000 lépés eredményét minden esetben letárljuk a memóriába
                         replay_memory.append([self.state, self.action, self.reward, new_state])
                     if len(replay_memory) >= 5000  and self.reward >= self.margin:       # az 5000 lépésen túl már csak a határértéken felül teljesítő lépéseket
                         replay_memory.append([self.state, self.action, self.reward, new_state])
                     #print("replay memory len ", len(replay_memory))
                     #print("replay memory ", replay_memory)
                                               
                     if self.steps_to_update_target_model % 5 == 0: # 5 lépésenként tanítjuk a hálót, a memóriából random eseteket visszajátszva
                        #print('\n training')
                        train(replay_memory, model, target_model)
                        #print("training end\n")
        
                     if self.steps_to_update_target_model % 100 == 0: # 100 lépés után a target háló súlyait is frissítjük, átmásoljuk a tanított alap hálóból
                        print('Copying main network weights to the target network weights')
                        target_model.set_weights(model.get_weights())
                        self.steps_to_update_target_model = 0
                        
                 self.state = new_state # állapot frissítése
                 self.steps_to_update_target_model += 1 # lépésszám növelése
                 self.episode_reward += self.reward # szumma reward aktualizálása
       
          
         # Ha a játék véget ért
         if fulljson["type"] == "leaderBoard":
             print("Game finished after",fulljson["payload"]["ticks"],"ticks!")
             print("Leaderboard:")
   
            # szerzett pontszám kiolvasása az adatcsomagból
             for score in fulljson["payload"]["players"]:
                 print(score["name"],score["active"], score["maxSize"])
                 if score["name"] == 'RemotePlayer':
                     self.total_score[self.episode-1]=score['maxSize']
                     self.sum_score += score['maxSize']
   
               
             time.sleep(1) # játék alaphelyzetbe állítása masterként

             mapran =  np.random.randint(1,4)  # random pályaválasztás
             if mapran == 0:
                 mapPath = "D:/BME/Msc/1. félév/Adaptív rendszerek modellezése/Porjekt/adaptivegame/src/maps/01_ring_empty.txt"
             elif mapran == 1:
                 mapPath = "D:/BME/Msc/1. félév/Adaptív rendszerek modellezése/Porjekt/adaptivegame/src/maps/02_base.txt"
             elif mapran == 2:
                 mapPath = "D:/BME/Msc/1. félév/Adaptív rendszerek modellezése/Porjekt/adaptivegame/src/maps/03_blockade.txt"
             else:
                 mapPath = "D:/BME/Msc/1. félév/Adaptív rendszerek modellezése/Porjekt/adaptivegame/src/maps/04_mirror.txt"
            
             fieldran =  np.random.randint(0,3) # random kajatérkép választás
             if fieldran == 0:
                fieldPath = "D:/BME/Msc/1. félév/Adaptív rendszerek modellezése/Porjekt/adaptivegame/src/fieldupdate/01_corner.txt"
             elif fieldran == 1:
                fieldPath = "D:/BME/Msc/1. félév/Adaptív rendszerek modellezése/Porjekt/adaptivegame/src/fieldupdate/02_cross.txt"
             else:
                fieldPath = "D:/BME/Msc/1. félév/Adaptív rendszerek modellezése/Porjekt/adaptivegame/src/fieldupdate/03_midlane.txt"
                
             #print(mapPath, fieldPath)
             sendData(json.dumps({"command": "GameControl", "name": "master",
                                  "payload": {"type": "reset", "data": {"mapPath": mapPath, "updateMapPath": fieldPath}}}))   
             
             # modellek mentése minden játk végén
             if Save_model_ON == True:
                 model.save(PATH+'/model')
                 target_model.save(PATH+'/target_model')
             
             
             self.epsilon = self.min_epsilon + (self.max_epsilon - self.min_epsilon) * np.exp(-self.decay * (self.episode)) # epsilon greedy frissítése
             
             self.total_rewards[self.episode-1]=self.episode_reward # kapott jutalom mentése
             self.sum_reward += self.episode_reward
             
             # az utolsó 10 játék összes rewardjából és elért pontszámából mozgó átlag számítása 
             if (self.episode) > 10:
                  self.sum_reward -= self.total_rewards[self.episode-10]
                  avg_reward = self.sum_reward/10
                  self.sum_score -= self.total_score[self.episode-10]
                  avg_score = self.sum_score/10
             else:
                  avg_reward = self.sum_reward/(self.episode)
                  avg_score = self.sum_score/(self.episode)
            
             self.total_avg_reward[self.episode-1]=avg_reward # mozgóátlagok letárolása
             self.total_avg_score[self.episode-1]=avg_score   

               # utolsó játék eredményeinek kiiratása
             print("\n")
             print("Episode: ", self.episode)
             print("Episode_reward: ", self.episode_reward)
             print("Episode_score: ", self.total_score[self.episode-1])
             print("AVG_reward: ", avg_reward)
             print("AVG_score: ", avg_score)
             print("epsilon: ", self.epsilon)
             print("len(replay_memory): ", len(replay_memory))
             #print("margin: ", self.margin)
             print("\n==========================================")
             print("\n")
             
             # kapott jutalom kirajzolása a játékok számának függvényében
             plt.title('REINFORCE Reward')
             plt.xlabel('Episode')
             plt.ylabel('Reward')
             plt.plot(self.total_rewards)
             plt.plot(self.total_avg_reward)
             plt.show()
             
             # elért pontszám kirajzolása a játékok számának függvényében
             plt.title('AI Score')
             plt.xlabel('Episode')
             plt.ylabel('Score')
             plt.plot(self.total_score)
             plt.plot(self.total_avg_score)
             plt.show()
             
             # néhány változó alaphelyzetbe állítása a játék végén
             self.episode_reward = 0
             self.size = 0
             self.oldsize = 5
              
                

if __name__=="__main__":
    
    # Példányosított stratégia objektum
    hunter = RemoteAI() 

    # neurális háló létrehozása vagy betöltése korábbi mentésből
    if Load_model_ON == False:
        model = create_model()
    else:
        model = keras.models.load_model(PATH+'/model')
    
    # target háló felépítése, és a súlyok bemásolása az alap hálóból
    target_model = create_model()
    target_model.set_weights(model.get_weights())


    # Socket kliens, melynek a szerver címét kell megadni (IP, port), illetve a callback függvényt, melynek szignatúrája a fenti
    client = SocketClient("localhost", 42069, hunter.processObservation)

    # Kliens indítása
    client.start()
    # Kis szünet, hogy a kapcsolat felépülhessen, a start nem blockol, a kliens külső szálon fut
    time.sleep(0.1)
    # Regisztráció a megfelelő névvel
    client.sendData(json.dumps({"command": "SetName", "name": "RemotePlayer", "payload": None}))

    # Nincs blokkoló hívás, a főszál várakozó állapotba kerül, itt végrehajthatók egyéb műveletek a kliens automata működésétől függetlenül.
    
#-----------------------------------------------------------------------------------------------------------------------
   

   

    