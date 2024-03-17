function [selected, generation, genAccuracy,bestAccuracy,bestStd,time] = WMFS_GA(X, Y, m,alpha)

%% Include dependencies
addpath('./lib'); % dependencies
addpath('./methods'); % FS methods
addpath(genpath('./lib/drtoolbox'));

sz = size(X);
for i = 1:sz(1)
    X{i} = normalizemeanstd(X{i});
%     X{i} = normalize(X{i}, 'range', [0 1]);
end
v = sz(1);


numF = 40;

% mrmr
for i = 1:v
    % mrmr
    index_mrmr{i} = mRMR(X{i}, Y, numF);
    fel_mrmr{i} = X{i}(:,index_mrmr{i}(:,1:end));
    acc_mrmr(i) = SVM(fel_mrmr{i},Y);
end

% calculate view weight
for i = 1:v
    v_weight(i) = acc_mrmr(i)/sum(acc_mrmr);
end

% calculate feature weight
for i = 1:v
    for j = 1:40
        fea_weight((i-1)*40+j) = v_weight(i)*(40-j+1)/40;
    end
end

init_index = [];
init_fea = [];
for i = 1:v
    init_index = [init_index,index_mrmr{i}];
    init_fea = [init_fea,fel_mrmr{i}];
end

%% GA
tic; % start timer

% 假设您有初始特征集 init_fea，其中包含 n 个样本和 d 个特征
d = length(init_index);
% 初始化连续最优个体未改变的计数器
consecutiveStagnation = 0;
consecutive = 0;
maxStagnationGenerations = 5;

% 遗传算法的参数
populationSize = 50; % 种群大小
numGenerations = 100; % 进化代数
tournamentSize = 5; % 锦标赛规模
mutationRate = 0.01; % 变异率
crossoverRate = 0.8; % 交叉概率
prevBestIndividual = [];

% 创建初始种群，每个个体包含 m 个特征
population = zeros(populationSize, d); % 初始化种群矩阵

for i = 1:populationSize
    % 随机选择 m 个特征的索引
    selectedIndices = randperm(d, m);
    % 将对应的特征位置设置为 1
    population(i, selectedIndices) = 1;
end

% 开始遗传算法的主循环
for generation = 1:numGenerations
    fprintf("generation %d \n",generation);
    % 计算每个个体的适应度值（这里使用评估指标的函数 evaluateFeatures）
    fitness = zeros(populationSize, 1);
    for i = 1:populationSize
        selectedFeatures = find(population(i, :));
        % 从 init_fea 中选择对应的特征列
        selectedData = init_fea(:, selectedFeatures);
        % 使用评估指标的函数 evaluateFeatures 评估性能
        [Accuracy(i),Std(i)] = SVM(selectedData,Y); % evaluateFeatures 是评估指标的函数
        W(i) = alpha*sum(fea_weight(selectedFeatures));
        fitness(i) = Accuracy(i) + W(i);
    end

    bestFitness = max(fitness); % 计算当前代的最优适应度
    bestIndividualIndex = find(fitness == bestFitness, 1);
    bestIndividual = population(bestIndividualIndex, :);
    genAccuracy(generation) = Accuracy(bestIndividualIndex);
    genStd(generation) = Std(bestIndividualIndex);
    % 检查是否连续若干代最优个体未改变
    if isequal(bestIndividual, prevBestIndividual)
        consecutiveStagnation = consecutiveStagnation + 1;
    else
        consecutiveStagnation = 0;
    end
    
    if consecutiveStagnation >= maxStagnationGenerations
        flag = 0;
        % 如果连续代数中最优个体未改变，结束算法
        fprintf('连续 %d 代最优个体未改变，提前结束算法。\n', maxStagnationGenerations);
        break;
    end

    prevBestIndividual = bestIndividual; % 更新前一个最优个体
    
    % 选择适应度高的个体作为父母
    parents = tournamentSelection(population, fitness,tournamentSize);
    
    % 使用交叉操作生成子代
    offspring = crossover(parents,m,crossoverRate);
    
    % 使用变异操作引入变异
    offspring = mutate(offspring, mutationRate);
    
    % 精英保留策略：将最优个体直接复制到下一代
    offspring(1, :) = bestIndividual; % 复制到下一代的第一个位置

    % 替换种群中的个体
    population = offspring;

end

time = toc; % stop timer and calculate elapsed time

% 返回最终的结果
selectedFeatures = find(bestIndividual(1, :));
selected = init_fea(:, selectedFeatures);
bestAccuracy = max(genAccuracy); % 计算当前代的最优适应度
bestAccuracyIndex = find(genAccuracy == bestAccuracy, 1);
bestStd = genStd(bestAccuracyIndex);
