TrainIp=table2array(Irradiationdata(1028:2927,4));                          % read data from workspace
TestIp=table2array(Generationdata(1005:2904,4));                           % read data from workspaceG
TestIp(isnan(TrainIp)) = [];                                                % remove NAN from DATA
TrainIp(isnan(TrainIp)) = [];                                               % remove NAN from DATA
TrainIp(TestIp>50)=[];                                                      % remove noise (more than 50) from DATA
TestIp(TestIp>50)=[];                                                       % remove noise (more than 50) from DATA
TrainIp(TestIp<0)=[];                                                       % remove noise (less than 0) from DATA
TestIp(TestIp<0)=[];                                                        % remove noise (less than 0) from DATA

TestIp(TrainIp<=0)=[];                                                      % remove noise (less than 0) from DATA
TrainIp(TrainIp<=0)=[];                                                     % remove noise (less than 0) from DATA

TrainIp=TrainIp';                                                           % convert row vs column
TestIp=TestIp';                                                             % convert row vs column
mn = min(TrainIp);                                                          % minimum of data
mx = max(TrainIp);                                                          % maximum of data
mn2 = min(TestIp);                                                          % minimum of data
mx2 = max(TestIp);                                                          % maximum of data

input = (TrainIp - mn) / (mx-mn);                                            %Normlize the Data
target = (TestIp - mn2) / (mx2-mn2);                                         %Normlize the Data
numTimeStepsTrain = floor(0.7*numel(TrainIp));                               % 70,10,20  percent training ,Validation and testing

XTrainIp = input(1:numTimeStepsTrain+1);                                     % training input data points
XTestIp = target(1:numTimeStepsTrain+1);                                     % training target data points
numFeatures = 2;                                                            % number of inputs=2
numResponses = 1;                                                           % number of output=1
numHiddenUnits = 200;                                                       % number of hidden unites

layers = [ ...
    sequenceInputLayer(numFeatures)
    lstmLayer(numHiddenUnits)
    fullyConnectedLayer(numResponses)
    regressionLayer];                                                       % LSTM layer structure

options = trainingOptions('adam', ...
    'MaxEpochs',250, ...
    'GradientThreshold',1, ...
    'InitialLearnRate',0.005, ...
    'MiniBatchSize',50, ...
    'LearnRateSchedule','piecewise', ...
    'LearnRateDropPeriod',90, ...
    'LearnRateDropFactor',0.2, ...
    'Verbose',false, ...
    'Plots','training-progress');                                           % LSTM other options
%     'ValidationData',{XTestIp,YTestIp},...
%     'ValidationFrequency',30, ...
net = trainNetwork([XTrainIp(1:end-1);XTestIp(1:end-1)],XTestIp(2:end),layers,options); % LSTM training
n=96
for k=1:1:4
  for j=1:(n/k)
  YTrainIp(1,j )=YTrainIp_sunny(1,k*j)
  YTestIp(1,j)=YTestIp_sunny(1,k*j)
end
[net,YPred] = predictAndUpdateState(net,[XTrainIp(end-1);XTestIp(end-1)]);  % LSTM prediction and update the network of last element of training data

numTimeStepsTest = numel(YTestIp);
for i = 2:numTimeStepsTest                                                  % LSTM prediction and update the network of next element of testing data
    [net,YPred(:,i)] = predictAndUpdateState(net,[YTrainIp(i-1);YPred(:,i-1)],'ExecutionEnvironment','cpu');
end                                                                         % predicted value is taken as input for the network (loop)

YPred = (mx2-mn2)*YPred + mn2;                                              % denormlize the predicted data as per min and max of target
YTest = YTestIp(1:end);
YTest = (mx2-mn2)*YTest + mn2;                                              % target data
rmse_s(1,k) = sqrt(mean((YPred-YTest).^2))                                         % error of network

XTestIp = (mx2-mn2)*XTestIp + mn2;     % denormlize the input data as per min and max of input
net = resetState(net);
net = predictAndUpdateState(net,[XTrainIp(1:end-1);XTestIp(1:end-1)]);      % train again
YPred = [];
numTimeStepsTest = numel(YTrainIp-1);
for i = 1:numTimeStepsTest                                                  % predict the output considerig new iputs in sequence
    [net,YPred(:,i)] = predictAndUpdateState(net,[YTrainIp(:,i);YTestIp(:,i)],'ExecutionEnvironment','cpu');
end
YPred = (mx2-mn2)*YPred + mn2;                                              % denormlize the predicted data as per min and max of target
rmse_s2(1,k) = sqrt(mean((YPred-YTest).^2))                                         % error of network
figure
subplot(2,1,1)
plot(YTest)
hold on
plot(YPred,'.-')
hold off
legend(["Observed" "Predicted"])
ylabel("Cases")
title("Forecast with Updates")

subplot(2,1,2)
stem(YPred - YTest)
xlabel("Month")
ylabel("Error")
title("RMSE = " + rmse_s2(1,k))
mae_s(1,k) = mean(abs(YPred-YTest));
mape_s(1,k)=mean(abs((YPred-YTest)/YTest))*100;

clear YTrainIp
clear YTestIp
clear YPred
clear YTest
clear numTimeStepsTest

end
YTrainIp_cloudy=(YTrainIp_cloudy - mn) / (mx-mn);
YTestIp_cloudy=(YTestIp_cloudy - mn2) / (mx2-mn2);
YTrainIp_cloudy=YTrainIp_cloudy';
YTestIp_cloudy=YTestIp_cloudy';


 for k=1:1:4
  for j=1:(n/k)
  YTrainIp(1,j )=YTrainIp_cloudy(1,k*j);
  YTestIp(1,j)=YTestIp_cloudy(1,k*j);
end
[net,YPred] = predictAndUpdateState(net,[XTrainIp(end-1);XTestIp(end-1)]);  % LSTM prediction and update the network of last element of training data

numTimeStepsTest = numel(YTestIp);
for i = 2:numTimeStepsTest                                                  % LSTM prediction and update the network of next element of testing data
    [net,YPred(:,i)] = predictAndUpdateState(net,[YTrainIp(i-1);YPred(:,i-1)],'ExecutionEnvironment','cpu');
end                                                                         % predicted value is taken as input for the network (loop)

YPred = (mx2-mn2)*YPred + mn2;                                              % denormlize the predicted data as per min and max of target
YTest = YTestIp(1:end);
YTest = (mx2-mn2)*YTest + mn2;                                              % target data
rmse_c(1,k) = sqrt(mean((YPred-YTest).^2));                                    % error of network


XTestIp = (mx2-mn2)*XTestIp + mn2;                  % denormlize the input data as per min and max of input
net = resetState(net);
net = predictAndUpdateState(net,[XTrainIp(1:end-1);XTestIp(1:end-1)]);      % train again
YPred = [];
numTimeStepsTest = numel(YTrainIp-1);
for i = 1:numTimeStepsTest                                                  % predict the output considerig new iputs in sequence
    [net,YPred(:,i)] = predictAndUpdateState(net,[YTrainIp(:,i);YTestIp(:,i)],'ExecutionEnvironment','cpu');
end
YPred = (mx2-mn2)*YPred + mn2;                                              % denormlize the predicted data as per min and max of target
rmse_c2(1,k) = sqrt(mean((YPred-YTest).^2)); % error of network
mae_c(1,k) = mean(abs(YPred-YTest));
mape_c(1,k)=mean(abs((YPred-YTest)/YTest))*100;
figure
subplot(2,1,1)
plot(YTest)
hold on
plot(YPred,'.-')
hold off
legend(["Observed" "Predicted"])
ylabel("Cases")
title("Forecast with Updates")

subplot(2,1,2)
stem(YPred - YTest)
xlabel("Month")
ylabel("Error")
title("RMSE = " + rmse_c2(1,k))

clear YTrainIp
clear YTestIp
clear YPred
clear YTest
clear numTimeStepsTest
end

 


