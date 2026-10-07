%% 车牌字符二分类器训练代码（数字 vs 非数字）
% 作者：AI助手
% 日期：2023-12-01

%% 步骤1：初始化环境
clear; clc; close all;
warning off images:imresize:sizeToInput

%% 步骤2：数据准备
% 数据目录结构：

dataPath = 'D:\char_data';  % 替换为你的实际路径

% 创建图像数据存储
imds = imageDatastore(dataPath, ...
    'IncludeSubfolders', true, ...
    'LabelSource', 'foldernames');

% 验证数据有效性
assert(numel(imds.Files) > 0, '未找到图像文件！检查路径设置');
assert(numel(unique(imds.Labels)) == 2, '必须包含digit和non_digit两个类别');

%% 步骤3：鲁棒的预处理函数
preprocessFcn = @(x) imresize(im2gray(im2uint8(x)), [32, 32]); % 统一尺寸和灰度化

% 验证预处理效果
sampleImg = preprocessFcn(readimage(imds, 1));
assert(isequal(size(sampleImg), [32 32]), '尺寸调整失败');
assert(ismatrix(sampleImg), '灰度转换失败');



% 显示预处理效果示例
sampleImg = preprocessFcn(readimage(imds, 1));
figure; 
subplot(1,2,1); imshow(readimage(imds, 1)); title('原始图像');
subplot(1,2,2); imshow(sampleImg); title('预处理后');
drawnow;

%% 步骤4：HOG特征提取

hogFeatureSize = 144; % 示例值，实际应根据图像尺寸计算
extractHOGFcn = @(x) extractHOGFeatures(preprocessFcn(x), ...
    'CellSize', [4 4], ...    % 调整单元格尺寸
    'BlockSize', [2 2], ...   % 调整块尺寸
    'NumBins', 9);            % 方向梯度数

% 初始化特征矩阵
numImages = numel(imds.Files);
hogFeatures = zeros(numImages, hogFeatureSize, 'single');
labels = imds.Labels;

% 特征提取主循环
hWaitbar = waitbar(0, '正在提取特征...');
% 将parfor改为普通for循环：
for i = 1:numImages
    img = readimage(imds, i);
    features(i, :) = extractHOGFcn(img);
    if mod(i,100)==0
        fprintf('已处理 %d/%d\n', i, numImages);
    end
end
close(hWaitbar);

%% 步骤5：划分训练集和测试集（80-20比例）
rng(1); % 固定随机种子保证可重复性
cv = cvpartition(labels, 'HoldOut', 0.2);

trainFeatures = features(cv.training, :);
trainLabels = labels(cv.training);
testFeatures = features(cv.test, :);
testLabels = labels(cv.test);

%% 步骤6：训练SVM分类器（带自动调参）
svmModel = fitcsvm(...
    trainFeatures, ...
    trainLabels, ...
    'KernelFunction', 'linear', ...    % 线性核适合高维特征
    'Standardize', true, ...           % 标准化数据
    'OptimizeHyperparameters', 'auto', ... % 自动优化参数
    'HyperparameterOptimizationOptions', struct(...
        'AcquisitionFunctionName', 'expected-improvement-plus', ...
        'ShowPlots', true, ...
        'Verbose', 1));

%% 步骤7：模型评估
[predLabels, scores] = predict(svmModel, testFeatures);

% 计算评估指标
accuracy = sum(predLabels == testLabels) / numel(testLabels);
confMat = confusionmat(testLabels, predLabels);
fprintf('测试准确率：%.2f%%\n', accuracy*100);
disp('混淆矩阵：');
disp(confMat);



%% 步骤8：可视化错误样本
testImages = imds.Files(cv.test);
misclassified = find(predLabels ~= testLabels);

figure;
for i = 1:min(9, numel(misclassified))
    subplot(3,3,i);
    img = preprocessFcn(readimage(imds, misclassified(i)));
    imshow(img);
    title(sprintf('预测: %d\n真实: %d', ...
        predLabels(misclassified(i)), ...
        testLabels(misclassified(i))));
end

%% 步骤9：保存模型（可选）
save('digitClassifier.mat', 'svmModel', 'preprocessFcn', 'extractHOGFcn');