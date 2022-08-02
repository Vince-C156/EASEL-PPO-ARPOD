import ARPOD_Gymenv
import matlab.engine
from ARPOD_Gymenv import HCW_ARPOD
u = matlab.double([0.0, 0.0, 0.0])

env = HCW_ARPOD([0.0, 0.0, 0.0, 1.0, 2.0, 0.0])
env.printstr("hello")

x = env.reset()
print(x)

env.step(u)
env.step(u)
