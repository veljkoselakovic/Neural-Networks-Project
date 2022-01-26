clc, clear, close all

%% Loading the data, this time from .csv

dat = readtable('JobChange.csv');

%% Isolate text to values and add it to the input
% This could have been done with a for loop but I think the time to write
% the loop would be bigger than just writing everything manually
input = [dat.city_development_index'];

[C,~,ib] = unique(dat.gender, 'stable');
input(2, :) = ib';

[C,~,ib] = unique(dat.relevent_experience, 'stable');
input(3, :) = ib';

[C,~,ib] = unique(dat.enrolled_university, 'stable');
input(4, :) = ib';

[C,~,ib] = unique(dat.education_level, 'stable');
input(5, :) = ib';

[C,~,ib] = unique(dat.major_discipline, 'stable');
input(6, :) = ib';

input(7, :) = dat.experience';

[C,~,ib] = unique(dat.company_size, 'stable');
input(8, :) = ib';

[C,~,ib] = unique(dat.company_type, 'stable');
input(9, :) = ib';

input(10, :) = dat.last_new_job';
input(11, :) = dat.training_hours';

%input

%% Create output
output = [dat.target'];

%% Histogram
figure
histogram(output);

%% Class division
K1 = input(:, output == 0);
K2 = input(:, output == 1);

% Can't plot this since we would need to have 11 dimensions

%% Test Training Validation sets

N1 = length(K1);
trainingK1 = K1(:, 1 : 0.7*N1);
testK1 = K1(:, 0.7*N1+1 : 0.85 * N1);
validationK1 = K1(:, 0.85*N1+1:N1);

N2 = length(K2);
trainingK2 = K2(:, 1 : 0.7*N2);
testK2 = K2(:, 0.7*N2+1 : 0.85 * N2);
validationK2 = K2(:, 0.85*N2+1:N2);

%%  Merged sets
training_input = [trainingK1, trainingK2];
training_output = [ones(1, round(0.7 * N1)), zeros(1, round(0.7*N2))];

idxs = randperm(length(training_input));
training_input = training_input(:, idxs);
training_output = training_output(idxs);

test_input = [testK1, testK2];
test_output = [ones(1, round(0.15 * N1)), zeros(1, round(0.15*N2)-1)];

validation_input = [validationK1, validationK2];
validation_output = [ones(1, round(0.15 * N1)), zeros(1, round(0.15*N2)-1)];

%% Final training sets
final_input = [training_input, test_input];
final_output = [training_output, test_output];

%% Krosvalidacija
arhitektura = {[10, 5], [12, 6, 3], [4]};
Abest = 0;
F1best = 0;

for reg = [0.1 : 0.1 : 1]
    for w = [1 : 1 : 15]
        for lr = [0.05, 0.005:0.1:0.5]
            for arh = length(arhitektura)
                rng(200);
                net = patternnet(arhitektura{arh});

                net.divideFcn = 'divideind';
                net.divideParam.trainInd = 1 : length(training_input);
                net.divideParam.valInd = length(training_input)+1 : length(final_input);
                net.divideParam.testInd = [];

                net.performParam.regularization = reg;

                net.trainFcn = 'traingda';

                net.trainParam.lr = lr;
                net.trainParam.epochs = 1000;
                net.trainParam.goal = 1e-4;
                net.trainParam.max_fail = 20;
                net.trainParam.showWindow = false;

                weight = ones(1, length(final_output));
                weight(final_output == 1) = w;

                [net, info] = train(net, final_input, final_output, [], [], weight);

                pred = sim(net, validation_input);
                pred = round(pred);

                [~, cm] = confusion(validation_output, pred);
                A = 100*sum(trace(cm))/sum(sum(cm));
                F1 = 2*cm(2, 2)/(cm(2, 1)+cm(1, 2)+2*cm(2, 2));

                disp(['Reg = ' num2str(reg) ', ACC = ' num2str(A) ', F1 = ' num2str(F1)])
                disp(['LR = ' num2str(lr) ', epoch = ' num2str(info.best_epoch)])

                if F1 > F1best
                    F1best = F1;
                    Abest = A;
                    reg_best = reg;
                    w_best = w;
                    lr_best = lr;
                    arh_best = arhitektura{arh};
                    ep_best = info.best_epoch;
                end
            end
        end
    end
end

%%Treniranje NM sa optimalnim parametrima (na celom trening + val skupu)
net = patternnet(arh_best);

net.divideFcn = '';

net.performParam.regularization = reg_best;

net.trainFcn = 'traingd';

net.trainParam.lr = lr_best;

net.trainParam.epochs = ep_best;
net.trainParam.goal = 1e-4;

weight = ones(1, length(final_output));
weight(final_output == 1) = w_best;

[net, info] = train(net, final_input, final_output, [], [], weight);

%% Performanse NM
pred = sim(net, test_input);
figure, plotconfusion(test_output, pred);
