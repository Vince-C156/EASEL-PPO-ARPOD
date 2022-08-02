% gravitational constant in km^2/s^2 from chance-constr MPC

classdef nonlinearChaserDynamics
    methods (Static)
        function traj = ChaserMotion(t,traj0,R,u)
            %{
                Constants:
                -----------
                    mu_GM: gravitational constant in km^2/s^2
                Paramters:
                ----------
                    traj0 = initial trajectory
                    R = orbital radius of the chaser spacecraft
                    u = external thrusters [ux,uy,uz]. modeled as a
                    function
                Returns:
                --------
                    [xdot,ydot,zdot,xdotdot,ydotdot,zdotdot] = HUGE matrix of 6 functions.
                    xdot,ydot,zdot = differential equation moved to be in terms of
                    first derivatives
        
                    xdotdot, ydotdot, zdotdot = differential equations moved to be
                    in terms of second derivatives
        
                    u know if i had half a mind, i'd go for zot, zotzot, and
                    zotzotzot hehe.
            %}
            %traj is [x,y,z,xdot,ydot,zdot]
            mu_GM = ARPOD_Benchmark.mu;
            x = traj0(1);
            y = traj0(2);
            z = traj0(3);
            
            ut = u(t); %function
            ux = ut(1); %indexing thrusters
            uy = ut(2);
            uz = ut(3);

            n = sqrt(mu_GM / (R.^3)); %orbital velocity
        
            %distance formula on chaser orbital radius ^3
            %resembles gravitational formula but generalized for 3d.
            const = ((R+x).^2 + y.^2 + z.^2).^(3/2); 
            
            xdot = traj0(4);
            ydot = traj0(5);
            zdot = traj0(6);
            xdotdot = 2*n*ydot + n*n*(R+x) - mu_GM*(R+x)/const + ux;
            ydotdot = -2*n*xdot + n*n*y - mu_GM*y/const + uy;
            zdotdot = -mu_GM*z/const + uz;
        
            %return
            traj = [xdot;ydot;zdot;xdotdot;ydotdot;zdotdot];
        end
        function [ts,trajs] = simulateMotion(traj0,R,u,T,timestep)
            f = @(t,traj) nonlinearChaserDynamics.ChaserMotion(t,traj,R,u);
            if timestep==0
                [ts,trajs] = ode45(f,[0,T], traj0);
            else
                [ts,trajs] = ode45(f,0:timestep:T, traj0);
            end
        end
        function jacobianMat = ChaserJacobian(t,traj0,R,u)
            mu_GM = nonlinearChaserDynamics.mu_GM;
            n = sqrt(mu_GM / (R.^3)); %orbital velocity
            x = traj0(1);
            y = traj0(2);
            z = traj0(3);

            jacobianMat = zeros(6,6);

            jacobianMat(1,4) = 1;
            jacobianMat(2,5) = 1;
            jacobianMat(3,6) = 1;

            norm = ( (R+x).^2 + y*y + z*z ).^(5/2);
            jacobianMat(4,1) = n*n - mu_GM*((-2*R*R)-4*R*x-2*x*x+y*y+z*z)/norm;
            jacobianMat(4,2) = mu_GM*(3*y*(R+x))/norm;
            jacobianMat(4,3) = mu_GM*(3*z*(R+x))/norm;
            jacobianMat(4,5) = 2*n;

            jacobianMat(5,1) = mu_GM*(3*y*(R+x))/norm;
            jacobianMat(5,2) = n*n - mu_GM*(R*R + 2*R*x + x*x - 2*y*y + z*z) / norm;
            jacobianMat(5,3) = mu_GM * 3*y*z/norm;
            jacobianMat(5,4) = -2*n;

            jacobianMat(6,1) = mu_GM*3*z*(R+x)/norm;
            jacobianMat(6,2) = mu_GM*3*y*z/norm;
            jacobianMat(6,3) = -mu_GM*(R*R+2*R*x+x*x+y*y-2*z*z)/norm;
        end
        function noisy_trajs = noisifyMotion(trajs, noise_model)
            [n_traj, dim_traj] = size(trajs);
            noisy_trajs = trajs;
            acc_noise = zeros([1,dim_traj]); %accumulated noise
            for i = 2:n_traj
                acc_noise = acc_noise + noise_model(); %make sure that system noise is consistent.
                noisy_trajs(i,:) = trajs(i,:) + acc_noise; 
            end
            %noisy_trajs is returned
        end
    end
end


