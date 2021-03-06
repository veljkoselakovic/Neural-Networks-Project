clc, clear, close all


%% Load the data

load('dataset2.mat');
rng(200);
    
input_ = pod(:, 1:2);
output_ = pod(:, 3);

%% Categorize the set
K1 = input_(output_==1, :);
K2 = input_(output_==2, :);
K3 = input_(output_==3, :);

output_OHE = zeros(length(output_), 3);
output_OHE(output_ == 1, 1 ) = 1;
output_OHE(output_ == 2, 2 ) = 1;
output_OHE(output_ == 3, 3 ) = 1;

%% Display the whole set

figure, hold all
plot(K1(:, 1), K1(:, 2), 'o')
plot(K2(:, 1), K2(:, 2), 'x')
plot(K3(:, 1), K3(:, 2), '*')
legend('K1', 'K2', 'K3');


%% Split the data into training and validation subsets
output_OHE = output_OHE';

N = length(input_);

TRAINING_COEFFICIENT = 0.9;
rng(200);
idxs = randperm(N);
idx_train = idxs(1 : TRAINING_COEFFICIENT * N);
idx_validation = idxs(TRAINING_COEFFICIENT * N + 1 : N);

training_input = input_(idx_train, :);
test_input = input_(idx_validation, :);

training_output = output_OHE(:, idx_train);
test_output = output_OHE(:, idx_validation);

%% Display the training vs validation subsets
training_input = training_input';
test_input = test_input';
%training_output = training_output';
%validation_output = validation_output';

figure, hold all
plot(training_input(1, :), training_input(2, :), 'ko');
plot(test_input(1, :), test_input(2, :), 'rx');
legend('Training', 'Test');






%% Create the neural network

LAYERS = {[4, 3, 3], [1,2,1], [9, 12, 6]};
EPOCHS = 800;
GOAL = 10e-4;
MIN_GRADIENT = 10e-6;
for i = 1 : length(LAYERS)


        rng(200);
        network = patternnet(LAYERS{i});

        network.performFcn = 'mse';
        network.divideFcn = '';


        network.trainParam.epochs = EPOCHS;
        network.trainParam.goal = GOAL;

        %network.layers{1}.transferFcn = 'poslin';   
        %network.layers{2}.transferFcn = 'poslin';
        %network.layers{3}.transferFcn = 'poslin';

        rng(200)
    %[training_input', training_output']
    network = train(network, training_input, training_output);

    %% Measuring performance

    display(network)
    network

    prediction = network(test_input);

    figure, plotconfusion(test_output, prediction);

    prediction = round(prediction)

    %% Decision areas

    Ntest = 500;
    ulazGO = [];
    x1 = linspace(-1, 1, Ntest);
    x2 = linspace(-1, 1, Ntest);

    for x11 = x1
        pom = [x11*ones(1, Ntest); x2];
        ulazGO = [ulazGO, pom];
    end

    predGO = sim(network, ulazGO);
    [vr, klasa] = max(predGO);

    K1go = ulazGO(:, klasa == 1);
    K2go = ulazGO(:, klasa == 2);
    K3go = ulazGO(:, klasa == 3);

    figure, hold all
    plot(K1go(1, :), K1go(2, :), '.')
    plot(K2go(1, :), K2go(2, :), '.')
    plot(K3go(1, :), K3go(2, :), '.')
    plot(K1(:, 1), K1(:, 2), 'bo')
    plot(K2(:, 1), K2(:, 2), 'r*')
    plot(K3(:, 1), K3(:, 2), 'yd')

    
    
end








