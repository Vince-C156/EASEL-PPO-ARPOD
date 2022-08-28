# EASEL-PPO-ARPOD
Using Proximal Policy Optimization to train a model for ARPOD missions

Python 3.8 or 3.9

Matlab version R2022a

https://www.mathworks.com/products/matlab/whatsnew.html

Must install matlab python engine before using

https://www.mathworks.com/help/matlab/matlab_external/install-the-matlab-engine-for-python.html

Matlab to Python version compatibility

https://www.mathworks.com/content/dam/mathworks/mathworks-dot-com/support/sysreq/files/python-compatibility.pdf

---------------------------------
Windows install instructions

-install matlab engine

cd '\Program Files\MATLAB\R2022a\extern\engines\python'

python setup.py install

*install on user profile only
python setup.py install --user 

-install matlab packages

control systems toolbox
Statistics and machine learning toolbox

---

*install requirements

pip install -r requirements.txt

if windows pip install -r requirements.txt does not work follow this tutorial

https://www.howtogeek.com/266621/how-to-make-windows-10-accept-file-paths-over-260-characters/
