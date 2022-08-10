import gym
from gym import spaces
import matlab.engine
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from collections import defaultdict


	
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

        self.action_space = spaces.Box(low=-6.0, high=6.0, shape=(3,), dtype=np.float64)
	# Example for using image as input (channel-first; channel-last also works):
        self.observation_space = spaces.Box(low=-10000.0, high=10000.0, shape=(6,), dtype=np.float64)

        self.prev_distance = self.distance_toTarget(self.observation)
        self.inital_distance = self.distance_toTarget(self.observation)

        self.inital_obstacle = ([500.0, 500.0, -500.0, -2, -3, -1], [200, 500, 300])
        self.obstacle_dict = {'obstacle_1' : self.inital_obstacle}

        self.info = defaultdict(list)
        self.done = False

    def step(self, action):
        self.time_elapsed += 1

        print("x0 IS ", self.x)
        #evolve system by t+1
        self.x = self.eng.ARPOD_Benchmark.nextStep(self.x, matlab.double(action), matlab.double(1), matlab.single(1),
                                                   nargout=1)[0]
        print("xdot IS ", self.x)
        self.observation = np.asarray(self.x, dtype=np.float64)[0]
        #evolve obstacles

        for obstacle_id, obstacle_data in self.obstacle_dict.items():

            state = obstacle_data[0]
            axes = obstacle_data[1]


            state_matlab = matlab.double(state)[0]
            
            evolved_state = self.eng.ARPOD_Benchmark.nextStep(state_matlab, matlab.double([0.0, 0.0, 0.0]), matlab.double(1), matlab.single(1), 
                                                              nargout=1)[0]

            evolved_state = np.asarray(evolved_state, dtype=np.float64)[0]
            self.obstacle_dict[obstacle_id] = (evolved_state, axes)


            chaserPos = self.observation[:3]
            obstaclePos = evolved_state[:3]
            obstacle_info = (obstaclePos, axes)

            self.info[obstacle_id].append(obstacle_info)

            if self.in_obstacle(chaserPos, obstacle_info):
                reward = -500
                self.done = True
                return self.observation, reward, self.done, self.info

        #check for out of bounds

        if not self.is_inbounds(self.observation):
            reward = -500
            self.done = True
            return self.observation, reward, self.done, self.info

        if self.time_elapsed >= 1000:
            reward = -300
            self.done = True
            return self.observation, reward, self.done, self.info

        #calculate reward


        distance = self.distance_toTarget(self.observation)

        if distance <= 50:
            self.done = True
            reward = 500
        else:
            reward = self.prev_distance - distance
        
        if reward <= 0:
            reward = -1

        self.prev_distance = distance

        print("REWARD ", reward)
        return self.observation, reward, self.done, self.info

    def distance_toTarget(self, obs):

        pos = obs[:3]
        x, y, z = pos[0], pos[1], pos[2]

        dist = ((x**2.0) + (y**2.0) + (z**2.0)) ** 0.5

        return dist


    def is_inbounds(self, obs):
        s = 3250.0
        a = (s**2.0 + s**2.0) ** 0.5
        b = s*2

        c = (a**2.0 + b**2.0) ** 0.5

        position = obs[:3]

        x, y, z = position[0], position[1], position[2]


        m_fromTarget = ((x**2.0) + (y**2.0) + (z**2.0)) ** 0.5

        if m_fromTarget >= c:
            return False
        else:
            return True    


    def in_obstacle(self, chaserPos, obstacle_data : tuple):

        """
        obstacle_data = (obstaclePos, [a, b, c])

        x is the chaser position and xc is the ellipsoid center position

        x = (x - xc)

        xT * P_inv * x >= 1 is true when point is outside of area and false
        if point is within area

        """

        obstaclePos = obstacle_data[0]
        obstacleAxes = np.asarray(obstacle_data[1], dtype=np.float64)

        obstacleAxes_squared = np.dot(obstacleAxes, obstacleAxes.T)

        x = chaserPos[0] - obstaclePos[0]
        y = chaserPos[1] - obstaclePos[1]
        z = chaserPos[2] - obstaclePos[2]

        X = np.asarray([[x],[y],[z]])
        XT = X.T


        P = np.zeros([3,3], dtype=np.float64)
        np.fill_diagonal(P, obstacleAxes_squared)

        print("P IS")
        print(P)
        P_inv = np.linalg.inv(P)

        XT_dot_P_inv =  np.dot(XT, P_inv)

        val = np.dot(XT_dot_P_inv, X)

        if val < 1:
            return True
        else:
            return False


    def random_x0(self):

        pos_mu = 500
        pos_sig = 50

        vel_mu = -3
        vel_sig = 1.7

        pos = pos_sig * np.random.randn(3,) + pos_mu
        vel = vel_sig * np.random.randn(3,) + vel_mu

        x0 = np.concatenate((pos, vel), axis=None)
        print(x0)

        return x0
    

    def reset(self):
        self.done=False

        print(self.x)
        print(type(self.x))
        self.time_elapsed = 0

        x0 = self.random_x0()
        self.x = matlab.double(x0)[0]
        self.observation = np.asarray(self.x, dtype=np.float64)[0]

        self.prev_distance = self.distance_toTarget(self.observation)

        self.info = defaultdict(list)
        self.obstacle_dict = {'obstacle_1' : self.inital_obstacle}

        return self.observation  # reward, done, info can't be included

    def render(self, mode='human'):
        pass
    def close (self):
        self.eng.quit()



  

