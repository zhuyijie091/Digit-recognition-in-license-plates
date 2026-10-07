
%% 批量车牌字符分割系统
clc; clear; close all;

% 配置路径
inputFolder = 'D:\大三\大三下\计算机视觉\课后\实验\实验二——车牌中的数字字符识别\data';
outputRoot = 'D:\大三\大三下\计算机视觉\课后\实验\实验二——车牌中的数字字符识别\char_samples';

% 获取所有图像文件
fileList = dir(fullfile(inputFolder, '*.jpg')); % 支持扩展名修改
numFiles = length(fileList);
fprintf('发现%d张待处理图像\n', numFiles);

% 通用参数设置
params = struct();
params.edgeThreshold = [0.01 0.25];    % Canny边缘检测阈值
params.morphSize = [3 15];             % 形态学操作结构元素尺寸
params.minArea = 1000;                 % 最小车牌区域面积
params.charSize = [32 32];             % 输出字符尺寸

%% 批量处理主循环
for i = 1:numFiles
    try
        %% 初始化
        tic;
        currentFile = fullfile(fileList(i).folder, fileList(i).name);
        [~, baseName] = fileparts(currentFile);
        outputFolder = fullfile(outputRoot, baseName);
        
        % 创建输出目录
        if ~exist(outputFolder, 'dir')
            mkdir(outputFolder);
        end
        
        fprintf('\n处理 %s (%d/%d)...\n', baseName, i, numFiles);
        
        %% 图像读取与预处理
        img = imread(currentFile);
        grayImg = rgb2gray(img);
        
        %% 车牌定位
        % 边缘检测
        edgeImg = edge(grayImg, 'Canny', params.edgeThreshold, 1.2);
        
        % 形态学操作
        se_horizontal = strel('rectangle', params.morphSize);
        morphImg = imclose(edgeImg, se_horizontal);
        se_vertical = strel('line', 5, 90);
        morphImg = imopen(morphImg, se_vertical);
        
        % 区域填充与筛选
        filledImg = imfill(morphImg, 'holes');
        filledImg = bwareafilt(filledImg, [params.minArea, 50000]);
        
        % 车牌区域检测
        stats = regionprops(filledImg, 'BoundingBox', 'Area', 'Extent', 'Solidity');
        plateBox = [];
        for k = 1:length(stats)
            bb = stats(k).BoundingBox;
            aspectRatio = bb(3)/bb(4);
            if aspectRatio > 2 && aspectRatio < 6 && ...
               stats(k).Extent > 0.1 && bb(3)*bb(4) > 10000
                plateBox = bb;
                break;
            end
        end
        
        %% 字符分割
        if isempty(plateBox)
            error('未检测到车牌区域');
        end
        
        % 提取车牌区域
        plateImg = imcrop(grayImg, plateBox);
        
        % 二值化与形态学处理
        plateBinary = imbinarize(plateImg, 'adaptive', 'Sensitivity', 0.7);
        plateClean = imopen(plateBinary, strel('rectangle', [1 3]));
        plateClean = bwareaopen(plateClean, 20);
        
        % 倾斜校正
        charStatsTemp = regionprops(plateClean, 'Orientation');
        if ~isempty(charStatsTemp)
            medianAngle = median([charStatsTemp.Orientation]);
            plateRotated = imrotate(plateClean, -medianAngle, 'bilinear', 'crop');
        else
            plateRotated = plateClean;
        end
        
        % 字符分割与筛选
        charStats = regionprops(plateRotated, 'BoundingBox', 'Image');
        allWidths = arrayfun(@(s) s.BoundingBox(3), charStats);
        if ~isempty(allWidths)
            medianWidth = median(allWidths);
            validChars = charStats(allWidths > medianWidth*0.6 & allWidths < medianWidth*2);
            xCoords = arrayfun(@(s) s.BoundingBox(1), validChars);
            [~, idx] = sort(xCoords);
            validChars = validChars(idx);
        else
            validChars = [];
        end
        
        %% 结果保存
        % 保存原始车牌图像
        imwrite(plateImg, fullfile(outputFolder, 'plate_original.png'));
        
        % 保存分割字符
        for k = 1:length(validChars)
            charImg = imresize(validChars(k).Image, params.charSize);
            imwrite(charImg, fullfile(outputFolder, sprintf('char_%02d.png', k)));
        end
        
        %% 生成处理报告
        fig = figure('Visible', 'off');
        subplot(2,2,1); imshow(img); title('原始图像');
        subplot(2,2,2); imshow(plateImg); title('定位车牌');
        subplot(2,2,3); imshow(plateClean); title('二值化处理');
        subplot(2,2,4); montage({validChars.Image}); title('分割字符');
        saveas(fig, fullfile(outputFolder, 'process_report.jpg'));
        close(fig);
        
        fprintf('处理成功! 保存%d个字符，耗时%.2fs\n', length(validChars), toc());
        
    catch ME
        fprintf('处理失败: %s\n', ME.message);
    end
end

fprintf('\n批量处理完成! 共成功处理%d/%d张图像\n', sum([fileList.processed]), numFiles);