N = 100;
x = [-1000,-2000,-3000,-100,-100,-100];
x_est = x.';
cov_t = eye(6);
tstep = 1;

trajTrue = zeros(N,6);
trajTrue(1,:) = x;

trajEst = zeros(N,6);
trajEst(1,:) = x;

noise = @() transpose(mvnrnd([0;0;0], [1,1,0.001], 1));

record_controlinput = zeros(3,N-1);
record_phases = zeros(1,N-1);

sensor_cov = 1000*[1,0,0;0,1,0;0,0,0.01];
process_cov = 1*eye(6);
for i = 2:N
    disp(i)
    phase = ARPOD_Benchmark.calculatePhase(x.', 0);
    record_phases(:,i-1) = phase;

    u_lqr = ChaserLQR.optimal_control(x_est, 1*eye(6), 10000*eye(3),ARPOD_Benchmark.a);

    %u_lqr = [5;5;5];
    record_controlinput(:,i-1) = u_lqr;
    x = ARPOD_Benchmark.nextStep(x,u_lqr,tstep, 1);
    trajTrue(i,:) = x.';

    %
    meas = ARPOD_Benchmark.sensor(x,noise,phase);
    if phase == 1
        [x_est,cov_t] = ChaserEKF.estimate(x_est, cov_t, u_lqr,tstep,ARPOD_Benchmark.a, meas, process_cov, sensor_cov(1:2,1:2), phase);
    else
        [x_est,cov_t]= ChaserEKF.estimate(x_est, cov_t, u_lqr,tstep,ARPOD_Benchmark.a, meas, process_cov, sensor_cov, phase);
    end
    trajEst(i,:) = x_est.';
end

figure(1)
plot3(trajEst(:,1), trajEst(:,2), trajEst(:,3), '-r');
hold on
plot3(trajTrue(:,1), trajTrue(:,2), trajTrue(:,3), '-b');
hold off
title('Benchmark')
xlabel('x')
ylabel('y')
zlabel('z')
grid on

figure(2)
plot(linspace(1,N,N),trajEst(:,1),'-r')
hold on
plot(linspace(1,N,N),trajTrue(:,1),'-b')
hold off
title('x trajectory over time')
grid on

figure(3)
plot(linspace(1,N-1,N-1),sum(record_controlinput.^2), '-b')
title('Control Input L2-Norm')
grid on

figure (4)
plot(linspace(1,N-1,N-1),record_phases,'-b')
title("Phase")