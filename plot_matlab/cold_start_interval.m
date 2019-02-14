% ==========================================================================
clear;
clc;

% 冷启动表
load data               % 
disp('Loading: data.mat');
lines = {
    %'d-', ...
    %'x-',  ...
    '<-', 's-', '*-', '^-'};
colors = {
    %[0 0 0.5], ...
    %[1 0.2 0],  ...
    [0 0 1],    [0 1 1],    [1 0 0],    [0 0.5 1]};
sub = [
    121, 122];
multiple = 100;
a = 't';       % 'taobao'拼写正确则为运行t3数据库，否则运行a1数据库

if strcmp(a, 't5')
    dataset = t5_test_interval;
    ylims = {       % y轴顶上留出空来，不然legend会压住刻度值
    [-130, 250]
    [-10, 45]};
else
    dataset = a5_test_interval;
    ylims = {       % y轴顶上留出空来，不然legend会压住刻度值
    [0.5, 50000]
    [-10, 65]};    
end

figure();
set(gca,'FontSize',20);
x = data_interval_idxs;
set(gca, 'XTick', x);   % 指定x轴刻度标识
    
for num = [1, 2]
    name = data_evaluation_growth_rate{num};    % 评价指标
    data = dataset{num};

    subplot(sub(num));
    for i = [1, 2, 3, 4]
        % 坐标轴用log，当绘制amazon里recall@30的增长率曲线时
        if ~strcmp(a, 't5') && strcmp(name, data_evaluation_growth_rate{1})   
            %  a5 - Recall@30 (%)，用log画。
            semilogy(x, data{i} * multiple, ...
                lines{i}, 'Color',colors{i}, 'LineWidth', 1.5, 'MarkerSize', 8);
        else
            % 其它情况，正常用(%)画。
            plot(x,data{i} * multiple, ...
                lines{i}, 'Color',colors{i}, 'LineWidth', 1.5, 'MarkerSize', 8);            
        end        
        hold on;    % 先画线，再hold on
    end
    
    xlabel('interval');
    labels = data_interval_labels;    
    set(gca, 'XTick', x, 'XTickLabel', labels);   % 指定x轴显示标识  
    xlim([0.5 10.5])
    ylabel(name)
    %ylim(ylims{num})
end

%hl = legend(data_method_growth_rate);       % 各种方法名
%set(hl, 'Location', 'NorthOutside', 'Orientation', 'horizontal','Box', 'on');





