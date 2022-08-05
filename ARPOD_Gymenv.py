import gym
from gym import spaces
import matlab.engine
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D



	
class HCW_ARPOD(gym.Env):
    """Custom Environment that follows gym interface"""

    def __init__(self, x0):
        eng = matlab.engine.start_matlab()
        super(HCW_ARPOD, self).__init__()
        self.eng = eng
        self.x0 = x0
        self.x = matlab.double(x0)[0]

        self.observation = np.asarray(self.x, dtype=np.float64)[0]
        self.time_elapsed = 0
        self.u = np.asarray([0.0, 0.0, 0.0], dtype=np.float64)
        #self.action = self.u

        self.lastcost = self.compute_cost(self.x, self.u)
        self.cost = None
        self.action_space = spaces.Box(low=-6.0, high=6.0, shape=(3,), dtype=np.float64)
	# Example for using image as input (channel-first; channel-last also works):
        self.observation_space = spaces.Box(low=-10000.0, high=10000.0, shape=(6,), dtype=np.float64)

        self.prev_distance = self.distance_toTarget(self.observation)
        self.inital_distance = self.distance_toTarget(self.observation)


    def step(self, action):
        self.time_elapsed += 1
        info = {}
        print("x0 IS ", self.x)
        #evolve system by t+1
        self.x = self.eng.ARPOD_Benchmark.nextStep(self.x, matlab.double(action), matlab.double(1), matlab.single(1),
                                                   nargout=1)[0]
        print("xdot IS ", self.x)

        self.observation = np.asarray(self.x, dtype=np.float64)[0]
        #check for out of bounds

        if not self.is_inbounds(self.observation):
            reward = -500
            self.done = True
            return self.observation, reward, self.done, info

        if self.time_elapsed >= 10000:
            reward = -200
            self.done = True
            return self.observation, reward, self.done, info

        #calculate reward


        distance = self.distance_toTarget(self.observation)

        if distance <= 50:
            self.done = True
            reward = 500
        else:
            reward = self.prev_distance - distance
        
        if reward <= 0:
            reward = 0

        self.prev_distance = distance

        print("REWARD ", reward)
        return self.observation, reward, self.done, info

    def distance_toTarget(self, obs):

        pos = obs[:3]
        x, y, z = pos[0], pos[1], pos[2]

        dist = ((x**2.0) + (y**2.0) + (z**2.0)) ** 0.5

        return dist

    def is_inbounds(self, obs):
        s = 2500.0
        a = (s**2.0 + s**2.0) ** 0.5
        b = s

        c = (a**2.0 + b**2.0) ** 0.5

        position = obs[:3]

        x, y, z = position[0], position[1], position[2]


        m_fromTarget = ((x**2.0) + (y**2.0) + (z**2.0)) ** 0.5

        if m_fromTarget >= c:
            return False
        else:
            return True    


    def printstr(self, x):
        print(x)

    def check_docked(self, x):
        x = np.asarray(x)[0]    
        position, velocities = x[:3], x[3:]
        posmag, velmag = np.linalg.norm(position), np.linalg.norm(velocities)
        
        if (posmag == 0) and (velmag == 0):
            return True
        else:
            print("not docked")
            return False
          
    def check_target_collision(self, x):
        x = np.asarray(x)[0]
        position, velocities = x[:3], x[3:]
        posmag, velmag = np.linalg.norm(position), np.linalg.norm(velocities) 
        print("positions ", position, " mag: ", posmag)
        print("velocities ", velocities, " mag: ", velmag)

        if (posmag == 0) and (velmag >=0):
            return True
        else:
            return False 

    def compute_cost(self, x, u):
        print("u param", u)
        print("x param", x)
        Q = np.asarray([[5.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                        [0.0, 5.0, 0.0, 0.0, 0.0, 0.0],
                        [0.0, 0.0, 5.0, 0.0, 0.0, 0.0],
                        [0.0, 0.0, 0.0, 10.0, 0.0, 0.0],
                        [0.0, 0.0, 0.0, 0.0, 10.0, 0.0],
                        [0.0, 0.0, 0.0, 0.0, 0.0, 0.0]])
        R = np.asarray(10.0)
        x = np.asarray(x, dtype=np.double)[0]
        u = np.asarray(u, dtype=np.double)
        print("x", x)
        xTQ = np.dot(x, Q)
        xTQx = np.dot(xTQ, x.T)
        print("u ", u)
        print("R", R)
        uTR = np.dot(u, R)
        uTRu = np.dot(uTR, u.T)
        print(xTQ)
        print("shape of u ", u.shape)
        print("shape of x ", x.shape)
        print("shape of R ", R.shape)
        cost = xTQx + uTRu
        print(cost)
        return cost 

    def reset(self):
        self.done=False

        print(self.x)
        print(type(self.x))
        self.time_elapsed = 0
        self.x = matlab.double(self.x0)[0]
        self.observation = np.asarray(self.x, dtype=np.float64)[0]
        return self.observation  # reward, done, info can't be included

    def render(self, mode='human'):
        pass
    def close (self):
        self.eng.quit()



  

