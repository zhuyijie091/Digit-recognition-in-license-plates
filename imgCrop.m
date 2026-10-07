% 读取图像并转换为灰度图
img = imread('D:\大三\大三下\计算机视觉\课后\实验\实验二——车牌中的数字字符识别\data\10.jpg');
grayImg = rgb2gray(img);

% 边缘检测增强轮廓
edgeImg = edge(grayImg, 'Canny', [0.01 0.25], 1.2); % 降低sigma减少噪声

% 形态学操作连接边缘
% 使用水平方向细长结构元素（原[5 25]改为[3 15]）
se_horizontal = strel('rectangle', [3 15]); 
morphImg = imclose(edgeImg, se_horizontal);

% 添加垂直方向开运算分离粘连区域
se_vertical = strel('line', 5, 90);
morphImg = imopen(morphImg, se_vertical);

% 填充区域并去除小对象
% 限制填充区域的最大面积（防止过度填充）
filledImg = imfill(morphImg, 'holes');
filledImg = bwareafilt(filledImg, [1000, 50000]); % 保留面积在1000-50000之间的区域

figure;
subplot(1,2,1); imshow(edgeImg); title('边缘检测');
subplot(1,2,2); imshow(filledImg); title('填充后区域');


% 定位候选区域
stats = regionprops(filledImg, 'BoundingBox', 'Area', 'Extent', 'Solidity');
fprintf('==== 候选区域统计信息 ====\n');
for k = 1:length(stats)
    fprintf('区域 %d:\n', k);
    fprintf('  边界框 [x,y,width,height] = [%.1f, %.1f, %.1f, %.1f]\n', stats(k).BoundingBox);
    fprintf('  面积 Area = %.1f\n', stats(k).Area);
    fprintf('  完整性 Extent = %.2f\n', stats(k).Extent);
    fprintf('  紧密度 Solidity = %.2f\n', stats(k).Solidity);
    fprintf('--------------------------\n');
end

% 根据特征筛选车牌区域
plateBox = [];
for k = 1:length(stats)
    bb = stats(k).BoundingBox;
    aspectRatio = bb(3)/bb(4);
    
    fprintf('检查区域 %d: BoundingBox = [%.1f, %.1f, %.1f, %.1f]\n', k, bb);
    
    if aspectRatio > 0.5 && aspectRatio < 6 && ...
       stats(k).Extent > 0.2 && bb(3)*bb(4) > 10000
        plateBox = bb;
        fprintf('>>> 找到候选车牌区域: [%.1f, %.1f, %.1f, %.1f]\n', bb);
        break;
    end
end

% 提取车牌区域
plateImg = imcrop(grayImg, plateBox);
figure;
imshow(plateImg);title('车牌区域');

% 优化车牌图像
plateBinary = imbinarize(plateImg, 'adaptive', 'Sensitivity', 0.7);

% 添加字符分离操作
se = strel('rectangle', [1 3]);  % 细长水平结构元素
plateClean = imopen(plateBinary, se);  % 开运算分离水平粘连

% 去除小噪声（面积阈值降低到20）
plateClean = bwareaopen(plateClean, 20);

% 步骤3：倾斜校正
charStatsTemp = regionprops(plateClean, 'Orientation');
angles = [charStatsTemp.Orientation];
medianAngle = median(angles);
plateRotated = imrotate(plateClean, -medianAngle, 'bilinear', 'crop');

% 步骤4：最终字符分割
charStats = regionprops(plateRotated, 'BoundingBox', 'Image');


% 动态宽度筛选
allWidths = arrayfun(@(s) s.BoundingBox(3), charStats);
if ~isempty(allWidths)
    medianWidth = median(allWidths);
    validIndices = find(allWidths > medianWidth*0.6 & ...
                      allWidths < medianWidth*2);
    validChars = charStats(validIndices);
else
    validChars = [];
end

% 按x坐标排序
if ~isempty(validChars)
    xCoords = arrayfun(@(s) s.BoundingBox(1), validChars);
    [~, idx] = sort(xCoords);
    validChars = validChars(idx);
end


% 保存每个字符
mkdir('D:\大三\大三下\计算机视觉\课后\实验\实验二——车牌中的数字字符识别\char_samples');
for k = 1:length(validChars)
    charImg = imresize(validChars(k).Image, [32 32]);
    imwrite(charImg, sprintf('char_samples/char_%02d.png', k));
end

% 可视化结果
figure;
subplot(2,2,1); imshow(img); title('原始图像');
subplot(2,2,2); imshow(plateImg); title('定位车牌');
subplot(2,2,3); imshow(plateClean); title('二值化处理');
subplot(2,2,4); montage({validChars.Image}); title('分割字符');