
%3DOF ARPOD sensing model
%assumes the trajectories are size 6 with position and their derivatives.
classdef ARPOD_Sensing
    methods (Static)
        function sense_data = convertTrajs(trajs)
            [n_traj, dim_traj] = size(trajs);
            e = zeros(3,n_traj);
            for i = 1:n_traj
                traj = trajs(i,:); 
                x = traj(1);
                y = traj(2);
                z = traj(3);
                norm = sqrt(x*x+y*y+z*z);
                e1 = atan(y/x);
                e2 = asin(z/norm);
                e3 = norm;
                e(:,i) = [e1;e2;e3];
            end
            sense_data = e;
        end
        function noisy_data = noisifyData(sense_data, noise_model)
            [dim_traj, n_traj] = size(sense_data);
            noisy_data = sense_data;
            for i = 1:n_traj
                noisy_data(:,i) = sense_data(:,i) + noise_model();
            end
            %noisy_trajs is returned
        end
        function z_t = measure(state) %add flags for measurement
            x = state(1);
            y = state(2);
            z = state(3);
            
            norm = sqrt(x*x+y*y+z*z);
            e1 = atan(y/x);
            e2 = asin(z/norm);
            e3 = norm;
            z_t = [e1;e2;e3];
        end
        function jacobian = jacobianMeasurement(x,y,z)
            jacobian = zeros(3,6);
            %dArctan
            partialX = -y/(x*x+y*y);
            partialY = x/(x*x+y*y);
            jacobian(1,1) = partialX;
            jacobian(1,2) = partialY;
            %rest of the partials are zero so it doesn't matter anyways.

            %dArcsin
            norm = sqrt( (x*x+y*y)/(x*x+y*y+z*z) ) * (x*x+y*y+z*z).^(3/2);
            partialX = -x*z/norm;
            partialY = -y*z/norm;
            partialZ = sqrt( (x*x+y*y)/(x*x+y*y+z*z) ) / sqrt(x*x+y*y+z*z);
            jacobian(2,1) = partialX;
            jacobian(2,2) = partialY;
            jacobian(2,3) = partialZ;

            %drho
            norm = sqrt(x*x+y*y+z*z);
            partialX = x/norm;
            partialY = y/norm;
            partialZ = z/norm;
            jacobian(3,1) = partialX;
            jacobian(3,2) = partialY;
            jacobian(3,3) = partialZ;

            %return jacobian
        end
        function jacobian = jacobianPartner(x,y,z,r)
            rx = r(1);
            ry = r(2);
            rz = r(3);

            jacobian = zeros(3,6);
            %dArctan
            partialX = (ry-y)/(ry*ry - 2*ry*y + rx*rx - 2*rx*x + x*x + y*y);
            partialY = (x-rx)/(ry*ry - 2*ry*y + rx*rx - 2*rx*x + x*x + y*y);
            jacobian(1,1) = partialX;
            jacobian(1,2) = partialY;
            %rest of the partials are zero so it doesn't matter anyways.

            %dArcsin
            bignorm = sqrt(1 - (rz-z).^2 / (rx-x).^2 + (ry-y).^2 + (rz-z).^2);
            norm = ((rx-x).^2 + (ry-y).^2 + (rz-z).^2).^(3/2) * bignorm;
            partialX = (rx-x)*(rz-z)/norm;
            partialY = (ry-y)*(rz-z)/norm;
            partialZ = (-rx*rx + 2*rx*x - ry*ry + 2*ry*y - rx*rx - ry*ry) / sqrt(x*x+y*y+z*z);
            jacobian(2,1) = partialX;
            jacobian(2,2) = partialY;
            jacobian(2,3) = partialZ;

            %drho
            norm = sqrt((rx-x).^2 + (ry-y).^2 + (rz-z).^2);
            partialX = -(rx-x)/norm;
            partialY = -(ry-y)/norm;
            partialZ = -(rz-z)/norm;
            jacobian(3,1) = partialX;
            jacobian(3,2) = partialY;
            jacobian(3,3) = partialZ;

            %return jacobian
        end
    end
end