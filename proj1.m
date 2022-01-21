clc, clear, close all


%% Load the data

load('dataset2.mat');
rng(50);
    
input_ = pod(:, 1:2);
output_ = pod(:, 3);

%% Categorize the set
K1 = input_(output_==1, :);
K2 = input_(output_==2, :);
K3 = input_(output_==3, :);

%% Display the whole set

figure, hold all
plot(K1(:, 1), K1(:, 2), 'o')
plot(K2(:, 1), K2(:, 2), 'x')
plot(K3(:, 1), K3(:, 2), '*')

%% Split the data into training and validation subsets

N = length(input_);

TRAINING_COEFFICIENT = 0.9;

idxs = randperm(N);
idx_train = idxs(1 : TRAINING_COEFFICIENT * N);
idx_validation = idxs(TRAINING_COEFFICIENT * N + 1 : N);

training_input = input_(idx_train, :);
validation_input = input_(idx_validation, :);

training_output = output_(idx_train);
validation_output = output_(idx_train);

%% Display the training vs validation subsets

figure, hold all
plot(training_input(:, 1), training_input(:, 2), 'ko');
plot(validation_input(:, 1), validation_input(:, 2), 'rx');


%% Create the neural network

LAYERS = [10, 20, 10];
EPOCHS = 500;
GOAL = 10e-20;
MIN_GRADIENT = 10e-6;

network = patternnet(LAYERS);

network.divideFcn = '';

network.trainParam.epochs = EPOCHS;
network.trainParam.goal = GOAL;
%network.trainParam.min_grad = MIN_GRADIENT;

%network.layers{1}.transferFcn = 'poslin';   
%network.layers{2}.transferFcn = 'poslin';
%network.layers{3}.transferFcn = 'poslin';

%% Train the network
view(network)
rng(50);
network = train(network, training_input', training_output');















