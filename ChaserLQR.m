classdef ChaserLQR
    properties (Constant)
    end
    methods (Static)
        function [A,B] = linearDynamics(R)
            mu_GM = 398600.4; %km^2/s^2;

            n = sqrt(mu_GM / (R.^3) );
            A = zeros(6,6);
            B = zeros(6,3);

            A(1:3,4:6) = eye(3);
            A(4,1) = 3*n*n;
            A(6,3) = -n*n;
            A(5,4) = -2*n;
            A(4,5) = 2*n;

            B(4:6,1:3) = eye(3);
        end
        function u = optimal_control(state, Q, R, R_orbit)
            % LQR does not care about any controller constraints imposed by
            % ARPOD problem. LQR should be used to help test the state
            % estimators.

            % cannot solve HJB formualtion since a matrix required needs to
            % solve Riccati Algebraic Equation

            [A,B] = ChaserLQR.linearDynamics(R_orbit);
            [K,S,e] = lqr(A,B,Q,R);
            u = -K*state;
        end
    end
end