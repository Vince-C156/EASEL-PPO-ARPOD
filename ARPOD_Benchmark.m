classdef ARPOD_Benchmark
    properties (Constant)
        t_e = 14400; % eclipse time (in seconds)
        t_f = 43200; % total mission duration (in seconds)
        rho_r = 1; % maximum distance for range measurements (1 km)
        %rho_r = 50;
        rho_d = 0.1; % docking phase initial radius (0.1 km)
        %rho_d = 25;
        m_t = 420000; % mass of target (2000 kg)
        m_c = 4200; % mass of chaser (500 kg)
        mu = 398600.4; %earth's gravitational constant (in km^3/s^2)
        a = 42164; % semi-major axis of GEO (42164 km)
        %a = 1000;
        Vbar = 5 * 10.^(-5); % max closing velocity while docking (in km/s)
        theta = 60; % LOS Constraint angle (in degrees)
        c = [-1;0;0]; % LOS cone direction
        x_docked = [0;0;0;0;0;0]; % docked position in km, and km/s
        x_relocation = [0;20;0;0;0;0]; %relocation position in km, km/s
        x_partner = [0;30;0;0;0;0]; %partner position in km, km/s

        % can choose to add noise separately
    end
    methods (Static)
        function phase = calculatePhase(traj, reached)
            norm = traj(1:3,:);
            norm = sqrt(norm.^2);
            if (reached == 0)
                if (norm > ARPOD_Benchmark.rho_r)
                    % ARPOD phase 1: Rendezvous w/out range
                    phase = 1;
                elseif (norm > ARPOD_Benchmark.rho_d) 
                    % ARPOD phase 2: Rendezvous with range
                    phase = 2;
                else 
                    %ARPOD phase 3: Docking
                    phase = 3;
                end
            else
                % ARPOD phase 4: Rendezvous to new location
                phase = 4;
            end
        end
        function traj = nextStep(traj0, u, timestep, options)
            fprintf('dimensions of u are\n')
            disp(size(u))
            disp(u)
            %fprintf("Converting\n")
            %u = cell2mat(u);
            disp(class(u))
            disp(u)
            if (options == 1)
                % discrete control input
                u0 = @(t) u;
                fprintf("TIMESET TYPE\n")
                disp(class(timestep))
                fprintf("PRINT U0\n")
                disp(u0)
                %u0 = cell2mat(u0);
                [ts, trajs] = nonlinearChaserDynamics.simulateMotion(traj0, ARPOD_Benchmark.a, u0, timestep, 0);
                traj = trajs(length(ts),:);
            elseif (options == 2)
                % discrete impulsive control input
                % To be Implemented
                disp('it is not implemented!. What are you doing?');
            elseif (options == 3)
                % continuous control input
                [ts, trajs] = nonlinearChaserDynamics.simulateMotion(traj0, ARPOD_Benchmark.a, u,timestep, 0);
                traj = trajs(length(ts),:);
            end
        end
        function sensor = sensor(state, noise, options)
            if (options == 1)
                %phase 1: only using bearing measurements
                sensor = ARPOD_Sensing.measure(state);
                sensor = sensor(1:2,:);
                w = noise();
                v = w(1:2,:);
            elseif (options == 2)
                %phase 2: bearing measurements + range measurement
                sensor = ARPOD_Sensing.measure(state);
                v = noise();
            elseif (options == 3)
                %phase 3: same as phase 2
                sensor = ARPOD_Sensing.measure(state);
                v = noise();
            elseif (options == 4)
                %phase 4: relative phase 2 to partner spacecraft
                r = ARPOD_Benchmark.x_partner - [state(1);state(2);state(3)]; % relative position to partner spacecraft
                sensor = ARPOD_Sensing.measure(r);
                v = noise();
            end
            sensor = sensor + v;
        end
        function jacobian = jacobianSensor(state, options, r)
            x = state(1);
            y = state(2);
            z = state(3);
            if (options == 1)
                jacobian = ARPOD_Sensing.jacobianMeasurement(x,y,z);
                jacobian = jacobian(1:2,:);
            elseif (options == 2)
                jacobian = ARPOD_Sensing.jacobianMeasurement(x,y,z);
            elseif (options == 3)
                jacobian = ARPOD_Sensing.jacobianMeasurement(x,y,z);
            elseif (options == 4)
                jacobian = ARPOD_Sensing.jacobianPartner(x,y,z,r);
            end
        end
    end
end