import gym
from gym import spaces
import matlab.engine
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from collections import defaultdict
import math
import random

	
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
        self.theta1 = 0.5
        self.theta2 = 0.5

        #action space between interval of -0.00002 to 0.00002, calculated with mass of chaser, 500kg, times a max of 10N from benchmark. 500 * 10 = 5000kg m/s^2
        #10N = 500kg * a (m/s^2)
        self.action_space = spaces.Box(low=-(1/50), high=(1/50), shape=(3,), dtype=np.float64)

	# Example for using image as input (channel-first; channel-last also works):
        self.observation_space = spaces.Box(low=-10.0, high=10.0, shape=(6,), dtype=np.float64)

        self.prev_distance = self.distance_toTarget(self.observation)
        self.inital_distance = self.distance_toTarget(self.observation)

        self.inital_obstacle = ([5.0, 2.0, -5.0, -0.002, 0.01, 0.0003], [0.00002, 0.00001, 0.00002])
        self.obstacle_dict = defaultdict(tuple)
        self.obstacle_dict['obstacle_1'] = self.inital_obstacle

        for i, obstacle in enumerate(self.random_obstacles(), start=2):
            print("ID VALUE")
            print(f'obstacle_[{i}]')
            self.obstacle_dict[f'obstacle_[{i}]'] = obstacle

        self.info = defaultdict(list)
        self.done = False

        """
        ending conditions: "obstacle_collision" "out_bounds" "step_limit" "docked"
        "current step"
        "steps in LOS"
        "x0"
        "step reward"
        "episode reward"
        "mission success"
        """

        self.episode_data = {"x0" : self.x0,
                             "last u": self.u,
                             "current step" : 0,
                             "steps in LOS" : 0,
                             "step reward" : 0,
                             "episode reward" : 0,
                             "ending condition" : "running",
                             "mission success" : False,
                             "ending state" : list()}

    def step(self, action):

        reward = 0
        self.time_elapsed += 1

        self.episode_data["current step"] = self.episode_data.get("current step") + 1
        self.episode_data["last u"] = action
        #evolve system by t+1
        self.x = self.eng.ARPOD_Benchmark.nextStep(self.x, matlab.double(action), matlab.double(1), matlab.single(1),
                                                   nargout=1)[0]
        self.observation = np.asarray(self.x, dtype=np.float64)[0]

        #current position no velocities
        chaserPos = self.observation[:3]

        #Terminal conditions

        #evolve obstacles
        for obstacle_id, obstacle_data in self.obstacle_dict.items():

            state = obstacle_data[0][:3]
            input_u = obstacle_data[0][3:]
            axes = obstacle_data[1]
            print("OBSTACLE STATE VEC", state, input_u )
            print(matlab.double(state))

            state_matlab = matlab.double(obstacle_data[0])[0]
            #u_matlab = matlab.double(np.asarray([0,0,0]))[0]

            
            evolved_state = self.eng.ARPOD_Benchmark.nextStep(state_matlab, matlab.double(np.asarray([0.0,0.0,0.0]))[0], matlab.double(1), matlab.single(1), 
                                                              nargout=1)[0]

            evolved_state = np.asarray(evolved_state, dtype=np.float64)[0]
            self.obstacle_dict[obstacle_id] = (evolved_state, axes)


            #chaserPos = self.observation[:3]
            obstaclePos = evolved_state[:3]
            obstacle_info = (obstaclePos, axes)

            self.info[obstacle_id].append(obstacle_info)

            """
            using function in_obstacle to check chaser position for collision within obstacle with
            ellipsoid constraint inequality.

            Terminal condition if true
            """

            if self.in_obstacle(chaserPos, obstacle_info):
                reward = -500
                self.episode_data["step reward"] = reward
                self.episode_data["ending condition"] = "obstacle_collision"
                self.episode_data["episode reward"] = self.episode_data.get("episode reward") + reward
                self.episode_data["ending state"].append(self.observation)

                self.done = True
                return self.observation, reward, self.done, self.info

        #check for out of bounds, Terminal condition if true
        if not self.is_inbounds(self.observation):
            reward = -500

            self.episode_data["step reward"] = reward
            self.episode_data["ending condition"] = "out_bounds"
            self.episode_data["episode reward"] = self.episode_data.get("episode reward") + reward
            self.episode_data["ending state"].append(self.observation)

            self.done = True
            return self.observation, reward, self.done, self.info

        #check for timelimit, terminal condition if true
        if self.time_elapsed >= 14400:
            reward = -300
            self.episode_data["step reward"] = reward
            self.episode_data["ending condition"] = "step_limit"
            self.episode_data["episode reward"] = self.episode_data.get("episode reward") + reward
            self.episode_data["ending state"].append(self.observation)
            self.done = True
            return self.observation, reward, self.done, self.info



        #Passive rewards

        """
        Testing if chaser is within a distance of the target, win condition if true

        If false episode continues with passive distance reward, lastdistance - currentdistance = reward
        if reward < 0, reward = -1
        """
        distance = self.distance_toTarget(self.observation)

        if distance <= 0.01:
            reward += 500
            self.episode_data["ending condition"] = "docked"
            self.episode_data["step reward"] = reward
            self.done = True
            self.episode_data["episode reward"] = self.episode_data.get("episode reward") + reward
            self.episode_data["mission success"] = True
            self.episode_data["ending state"].append(self.observation)
            return self.observation, reward, self.done, self.info
        else:
            #reward += self.prev_distance - distance
            initial_distance = self.distance_toTarget(self.episode_data["x0"])
            progress = initial_distance - distance
            print(f'PROGRESS {progress}')
        
        if progress <= 0:
            reward += -2
        else:
            r = progress * 1.5
            reward += r

        #self.prev_distance = distance


        """
        Testing of chaser is within the LOS pyramid region constraint
        """

        if self.in_LOS(chaserPos):
            reward += 15
            self.episode_data.get("steps in LOS") + 1
            print("IN LOS")
        else:
            print("OUTSIDE LOS")
            reward += -0.5

        self.episode_data["step reward"] = reward
        self.episode_data["episode reward"] = self.episode_data.get("episode reward") + reward

        return self.observation, reward, self.done, self.info

    def distance_toTarget(self, obs):
        """
        Using distance formula with chaser state positions to calculate length between the two spacecrafts
        """

        pos = obs[:3]
        x, y, z = pos[0], pos[1], pos[2]

        dist = ((x**2.0) + (y**2.0) + (z**2.0)) ** 0.5

        return dist


    def is_inbounds(self, obs):
        """
        Calculating a domain that equally extends slightly over 10km away from the chaser.
        """
        s = 10
        a = (s**2.0 + s**2.0) ** 0.5
        b = s*2

        c = (a**2.0 + b**2.0) ** 0.5

        position = obs[:3]

        x, y, z = position[0], position[1], position[2]


        km_fromTarget = ((x**2.0) + (y**2.0) + (z**2.0)) ** 0.5

        if km_fromTarget >= c:
            return False
        else:
            return True    


    def in_target(self):
        pass

    def target_collision(self):
        pass
    
    def in_obstacle(self, chaserPos, obstacle_data : tuple):

        """
        obstacle_data = (obstaclePos, [a, b, c])

        x is the chaser position and xc is the ellipsoid center position

        x = (x - xc)

        xT * P_inv * x >= 1 is true when point is outside of area and false
        if point is within area

        """

        obstaclePos = obstacle_data[0]
        obstacleAxes = np.asarray(obstacle_data[1], dtype=np.dtype('d'))

        obstacleAxes_squared = np.dot(obstacleAxes, obstacleAxes.T)

        x = chaserPos[0] - obstaclePos[0]
        y = chaserPos[1] - obstaclePos[1]
        z = chaserPos[2] - obstaclePos[2]

        X = np.asarray([[x],[y],[z]])
        XT = X.T


        P = np.zeros([3,3], dtype=np.dtype('d'))
        np.fill_diagonal(P, obstacleAxes_squared)

        P_inv = np.linalg.inv(P)

        XT_dot_P_inv =  np.dot(XT, P_inv)

        val = np.dot(XT_dot_P_inv, X)

        if val < 1:
            return True
        else:
            return False

    def in_LOS(self, chaserPos):
        """
        Testing if chaser point satisfies pyramid constraint inequality formulation. The region is a pyramid bounded by four planes along the x axis.
        switch x and z positions if desired LOS points up or down.

        Ax <= b

        A : [[sin(theta1/2) cos(theta1/2), 0],
             [sin(theta1/2) -cos(theta1/2), 0],
             [sin(theta2/2) 0, cos(theta2/2)],
             [sin(theta2/2) 0, -cos(theta2/2)],
                                              ]
        x : [[x],
             [y],
             [z],]

        b : [[0],
             [0],
             [0],
             [0]]

        1. np matrix A
        2. pass current state position
        3. return true or false
        """

        A = np.asmatrix([[np.sin(self.theta1/2), np.cos(self.theta1/2), 0], 
              [np.sin(self.theta1/2), -np.cos(self.theta1/2), 0],
              [np.sin(self.theta2/2), 0, np.cos(self.theta2/2)],
              [np.sin(self.theta2/2), 0, -np.cos(self.theta2/2)]])

        x = np.asmatrix([ [chaserPos[0],chaserPos[1],chaserPos[2]] ]).T

        b = np.asmatrix([ [0.0, 0.0, 0.0, 0.0] ]).T

        Ax = A @ x

        values = np.less_equal(Ax, b)


        """
        returns True if all inequalities evaluate to true otherwise false
        """
        if np.all(values):
            return True
        else:
            return False


    def random_x0(self):
        """
        Method that returns a random initalization point for the start of an episode within a range
        that is far from the target
        """

        pos_mu = -9.5
        pos_sig = 3.0

        vel_mu = -2.0
        vel_sig = 1.41421356237

        pos = pos_sig + pos_mu * np.random.randn(3,).astype(np.float64)
        vel = vel_sig  + vel_mu * np.random.randn(3,).astype(np.float64)

        vel /= np.float64(1000.0)

        x0 = np.concatenate((pos, vel), axis=None)
        print(x0)

        return x0

    def random_obstacles(self):
        """
        Method that returns a random initalization point for obstacles a range
        that is medium to short distance from target
        """

        new_obstacles = list()

        n = random.randint(0, 3)

        sqrthigh_pos = 2.5
        low_pos = -8.0

        sqrthigh_vel = 1.41421356237
        low_vel = -2.0

        sqrtdim_high = 2.5
        dim_low = -3.0

        for i in range(0, n):
            pos = sqrthigh_pos + low_pos * np.random.randn(3,)
            vel = sqrthigh_vel + low_vel * np.random.randn(3,)

            a, b, c = sqrtdim_high + dim_low * np.random.randn(3,)

            a /= np.float64(1000.0)
            b /= np.float64(1000.0)
            c /= np.float64(1000.0)


            #a, b, c = int(a), int(b), int(c)

            state = np.concatenate((pos, vel), axis=None, dtype=np.float64)
            axes = [a, b, c]

            new_obstacles.append((state, axes))

        return new_obstacles
            

        
    

    def reset(self):
        """
        Method that resets all parameters and generates new spawn and obstacle instances for the
        next episode
        """
        self.done=False

        print(self.x)
        print(type(self.x))
        self.time_elapsed = 0

        x0 = self.random_x0()
        self.x = matlab.double(x0)[0]
        self.observation = np.asarray(self.x, dtype=np.float64)[0]

        self.prev_distance = self.distance_toTarget(self.observation)

        self.info = defaultdict(list)
        self.obstacle_dict = defaultdict(tuple)
        self.obstacle_dict['obstacle_1'] = self.inital_obstacle

        for i, obstacle in enumerate(self.random_obstacles(), start=2):
            print("ID VALUE")
            print(f'obstacle_[{i}]')
            self.obstacle_dict[f'obstacle_[{i}]'] = obstacle

        self.episode_data = {"x0" : self.observation,
                             "last u": self.u,
                             "current step" : 0,
                             "steps in LOS" : 0,
                             "step reward" : 0,
                             "episode reward" : 0,
                             "ending condition" : "running",
                             "mission success" : False,
                             "ending state" : list()}

        return self.observation  # reward, done, info can't be included

    def render(self, mode='human'):
        pass
    def close (self):
        self.eng.quit()



  

