% ==========================================================================
clear;
clc;

% 维度表
load data               % 
disp('Loading: data.mat');
lines = {
    '>:', 'p--', 'd--', ...
    'o--', '*--', ...
    '<-', 's-', 'x-', '^-'};
    % :  Dotted line
    % -. Dash-dotted line
    % -- Dashed line
    % -  Solid line
colors = {
    [0, 0, 0],  [1 0 1],      [0 0 0.5], ...
    [0 0.5 0],  [1 0.2 0],  ...
    [0 0 1],    [0 1 1],    [1 0 0],    [0 0.5 1]};
sub = [
    141, 142, 143, 144];
a = 't';       % 'taobao'拼写正确则为运行t3数据库，否则运行a1数据库
if strcmp(a, 't5')
    dataset = t5_valid_weidu;
    ylims = {       % y轴顶上留出空来，不然legend会压住刻度值
    [0.55, 1.55]
    [0.1, 0.45]
    [0.2, 0.9]
    [0.5, 0.7]};
else
    dataset = a5_valid_weidu;
    ylims = {       % y轴顶上留出空来，不然legend会压住刻度值
    [0.2, 0.95]
    [0.04, 0.195]
    [0.1, 0.48]
    [0.35, 0.8]};    
end

figure();
set(gca,'FontSize', 20);
x = data_dimensionality;
set(gca, 'XTick', x);   % 指定x轴刻度标识
    
for num = [1, 2, 3, 4]
    name = data_evaluation{num};
    data = dataset{num};

    subplot(sub(num));
    for i = [1, 2, 3, 4, 5, 6, 7, 8, 9]
        plot(x,data{i}, lines{i}, 'Color',colors{i}, 'LineWidth', 2, 'MarkerSize', 8);
        hold on;        
    end
    
    xlabel('dimension');
    xlim([9 26])
    ylabel(name)
    %ylim(ylims{num})
end

%h1 = legend(data_method);
%set(h1, 'Location', 'NorthOutside', 'Orientation', 'horizontal', 'Box', 'on');


