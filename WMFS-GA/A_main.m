%% DEMO FILE
clc;clear;close all
% Include dependencies
addpath('./data/');
addpath('./res');

%% Load the data and select features for classification
load('MSRCV1.mat');

%% 定义 alpha 参数范围
% alpha_grid = [0.001, 0.01, 0.1, 1, 10, 100, 1000];
alpha_grid = [0];
m=40;

% 设置保存结果的目录
saveDir = './res/';

for i = 1:length(alpha_grid)
        % 在这里运行您的遗传算法代码，修改 alpha 参数为当前迭代的值
        alpha = alpha_grid(i);
        % 模拟运行，实际需要将下面的代码替换为您的遗传算法运行代码
        [selected, generation, genAccuracy,bestAccuracy,bestStd,time] = WMFS_GA(X, Y, m,alpha);
        [accuracy,std] = SVM(selected,Y);
        % 记录结果
        result = struct('alpha', alpha, 'selected', selected, 'generation', generation, 'genAccuracy', genAccuracy ...
            ,'bestAccuracy',bestAccuracy,'std',bestStd,'time',time);
%         results = [results, result];
end

%% 绘制曲线图1：参数 vs. 迭代次数
figure;
plot([results.alpha], [results.iterations], '-*','LineWidth',1.5);
xlabel('Alpha');
ylabel('Generations');
title('Generations vs. Alpha');
saveas(gcf, [saveDir, 'iterations_vs_alpha.png']);

% 绘制曲线图2：参数 vs. 准确度
figure;
plot([results.alpha], [results.bestAccuracy], '-o','LineWidth',1.5,'Color',[0.85,0.33,0.10]);
xlabel('Alpha');
ylabel('Accuracy');
title('Accuracy vs. Alpha');
saveas(gcf, [saveDir, 'accuracy_vs_alpha.png']);

%% 找到准确度最高的结果
[maxAccuracy, maxIdx] = max([results.bestAccuracy]);
bestResult = results(maxIdx);
% 保存最优个体的特征选择到文件
bestIndividual = bestResult.bestIndividual;
% [acc,std] = SVM(bestIndividual,Y);
% fprintf('acc x 100 ± std x 100: %.2f±%.2f\n', acc*100, std*100);
save([saveDir, 'best_individual.mat'], 'bestIndividual');


%% 绘制曲线图3：最优参数对应的迭代次数 vs. 准确度
figure;
plot(1:generation, genAccuracy, 'LineWidth',1.5,'Color',[0.85,0.33,0.10]);
xlabel('Generations');
ylabel('Accuracy');
title('Accuracy vs. Generations');
saveas(gcf, [saveDir, 'msrcv1_accuracy_vs_generations.png']);


