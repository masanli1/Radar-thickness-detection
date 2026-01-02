clear; clc; close all;

%% 第一部分：系统参数设置与真实数据加载
% =========================================================================
fprintf('=== 真实雷达数据厚度测量 - 基于TI IWR6843配置 ===\n');

% 1.1 雷达系统参数（基于您提供的配置文件）
c = 299792458;          % 光速 (m/s)

% 从配置文件中提取的参数：
% chirpTimingCfg: ChirpRfFreqStart = 77 GHz
f0 = 77e9;              % 起始频率 77 GHz

% chirpTimingCfg: ChirpRfFreqSlope = 88 MHz/μs
slope_MHz_per_us = 88;  % MHz/μs
slope = slope_MHz_per_us * 1e12;  % Hz/s (88 MHz/μs = 88e12 Hz/s)

% chirpComnCfg: NumOfAdcSamples = 512
N_per_chirp = 512;      % 每个啁啾的采样点数

% chirpComnCfg: DigOutputSampRate = 8
% 根据TI文档，这个值对应采样率：100/8 = 12.5 MHz
fs = 12.5e6;            % 采样率 12.5 MHz

% chirpTimingCfg: ChirpRampEndTime = 45 μs
Tc = 45e-6;             % 啁啾斜坡时间 45 μs

% 参数计算
% 实际采样时间 = 采样点数 / 采样率
actual_sampling_time = N_per_chirp / fs;  % 512 / 12.5e6 = 40.96 μs
% 带宽 = 斜率 × 实际采样时间
BW = slope * actual_sampling_time;

fprintf('雷达系统参数 (基于配置文件):\n');
fprintf('  起始频率: %.2f GHz\n', f0/1e9);
fprintf('  斜率: %.0f MHz/μs = %.2e Hz/s\n', slope_MHz_per_us, slope);
fprintf('  采样率: %.2f Msps\n', fs/1e6);
fprintf('  每啁啾采样点数: %d\n', N_per_chirp);
fprintf('  啁啾斜坡时间: %.1f μs\n', Tc*1e6);
fprintf('  实际采样时间: %.2f μs\n', actual_sampling_time*1e6);
fprintf('  带宽: %.2f GHz\n', BW/1e9);
fprintf('  距离分辨率: %.2f mm\n', c/(2*BW)*1000);

% 1.2 系统配置参数（来自配置文件）
% channelCfg: RxChCtrlBitMask=7 (二进制111，表示3个RX通道)
%             TxChCtrlBitMask=3 (二进制011，表示2个TX通道)
n_rx_chan = 3;          % 接收通道数
n_tx_chan = 2;          % 发射通道数
virtual_channels = n_rx_chan * n_tx_chan;  % 虚拟通道数 = 6

% frameCfg: NumOfChirpsInBurst=64, NumOfBurstsInFrame=1, NumOfFrames=10
n_chirps_per_frame = 64;  % 每帧啁啾数
n_frames = 10;           % 总帧数

fprintf('系统配置:\n');
fprintf('  接收通道数: %d\n', n_rx_chan);
fprintf('  发射通道数: %d\n', n_tx_chan);
fprintf('  虚拟通道数: %d\n', virtual_channels);
fprintf('  每帧啁啾数: %d\n', n_chirps_per_frame);
fprintf('  总帧数: %d\n', n_frames);

% 1.3 时间向量（单个啁啾）
t_chirp = (0:N_per_chirp-1)' / fs;  % 单个啁啾的时间向量 (s)

% 1.4 加载真实雷达数据
fprintf('\n加载真实雷达数据...\n');

% 尝试从二进制文件加载数据
[filename, pathname] = uigetfile({'*.bin;*.mat;*.csv', '雷达数据文件'});
if isequal(filename, 0)
    error('未选择数据文件');
end

real_data_file = fullfile(pathname, filename);
[~, ~, ext] = fileparts(filename);

% 根据文件扩展名选择加载方式
if strcmpi(ext, '.bin')
    % 读取二进制数据（假设为16位整数，I/Q交织）
    fid = fopen(real_data_file, 'rb');
    if fid == -1
        error('无法打开数据文件: %s', real_data_file);
    end
    
    raw_data = fread(fid, inf, 'int16');
    fclose(fid);
    
    fprintf('二进制数据加载成功\n');
    fprintf('  原始16位整数点数: %d\n', length(raw_data));
    
    % 转换为复数（假设I/Q交织）
    if mod(length(raw_data), 2) == 0
        I_data = raw_data(1:2:end);
        Q_data = raw_data(2:2:end);
        real_data_complex = complex(I_data, Q_data);
        
        % 归一化到[-1, 1]范围
        real_data_complex = real_data_complex / 32767;
        
        fprintf('  复数点数: %d\n', length(real_data_complex));
    else
        error('数据长度不是偶数，无法解析为I/Q数据');
    end
    
elseif strcmpi(ext, '.mat')
    % 从MAT文件加载数据
    data_struct = load(real_data_file);
    
    % 尝试找到复数数据
    if isfield(data_struct, 'radar_data')
        real_data_complex = data_struct.radar_data;
    elseif isfield(data_struct, 'data')
        real_data_complex = data_struct.data;
    elseif isfield(data_struct, 'if_signal')
        real_data_complex = data_struct.if_signal;
    else
        % 尝试找到第一个复数数组
        field_names = fieldnames(data_struct);
        found = false;
        for i = 1:length(field_names)
            if isa(data_struct.(field_names{i}), 'double') || isa(data_struct.(field_names{i}), 'single')
                if ~isreal(data_struct.(field_names{i}))
                    real_data_complex = data_struct.(field_names{i});
                    found = true;
                    fprintf('找到复数数据字段: %s\n', field_names{i});
                    break;
                end
            end
        end
        
        if ~found
            error('未找到复数数据');
        end
    end
    
    fprintf('MAT文件数据加载成功\n');
    
elseif strcmpi(ext, '.csv')
    % 从CSV文件加载数据
    csv_data = readmatrix(real_data_file);
    
    % 假设第一列是实部，第二列是虚部
    if size(csv_data, 2) >= 2
        real_data_complex = complex(csv_data(:,1), csv_data(:,2));
        fprintf('CSV文件数据加载成功\n');
    else
        error('CSV文件需要至少两列数据（实部和虚部）');
    end
end

% 1.5 数据预处理 - 根据雷达配置解析数据结构
fprintf('\n数据预处理...\n');

% 计算每帧的复数点数
complex_samples_per_frame = N_per_chirp * n_chirps_per_frame * virtual_channels;
fprintf('每帧的复数点数: %d\n', complex_samples_per_frame);

% 计算总帧数
total_complex_samples = length(real_data_complex);
num_frames_actual = floor(total_complex_samples / complex_samples_per_frame);

if num_frames_actual < 1
    error('数据量不足一帧，需要至少 %d 个复数点', complex_samples_per_frame);
end

fprintf('检测到 %d 帧数据\n', num_frames_actual);
fprintf('总复数点数: %d\n', total_complex_samples);

% 重塑数据为4维数组: [采样点, 啁啾, 虚拟通道, 帧]
% 截取完整帧的数据
valid_samples = num_frames_actual * complex_samples_per_frame;
real_data_complex = real_data_complex(1:valid_samples);

radar_data_4d = reshape(real_data_complex, ...
    N_per_chirp, n_chirps_per_frame, virtual_channels, num_frames_actual);

fprintf('数据重塑为: %d采样点 × %d啁啾 × %d虚拟通道 × %d帧\n', ...
    size(radar_data_4d, 1), size(radar_data_4d, 2), size(radar_data_4d, 3), size(radar_data_4d, 4));

% 1.6 选择处理的数据
% 可以选择单帧或多帧平均
use_frame_average = true;  % 是否使用多帧平均
frame_to_process = 1;      % 如果不用平均，处理哪一帧

if use_frame_average && num_frames_actual > 1
    % 对多个帧进行平均，提高信噪比
    real_data_mean = mean(radar_data_4d, 4);
    
    % 然后对啁啾进行平均（假设目标静止）
    real_data_mean_chirp = mean(real_data_mean, 2);  % 平均所有啁啾
    real_data_mean_chirp = squeeze(real_data_mean_chirp);  % 移除啁啾维度
    
    % 对虚拟通道进行平均
    real_if_signal = mean(real_data_mean_chirp, 2);  % 平均所有虚拟通道
    
    fprintf('使用多帧平均: %d帧 + %d啁啾 + %d通道\n', ...
        num_frames_actual, n_chirps_per_frame, virtual_channels);
else
    % 使用单帧数据
    frame_data = radar_data_4d(:, :, :, frame_to_process);
    
    % 对啁啾进行平均
    frame_data_mean_chirp = mean(frame_data, 2);
    frame_data_mean_chirp = squeeze(frame_data_mean_chirp);
    
    % 对虚拟通道进行平均
    real_if_signal = mean(frame_data_mean_chirp, 2);
    
    fprintf('使用单帧数据: 第%d帧\n', frame_to_process);
end

% 确保数据长度为N_per_chirp
if length(real_if_signal) ~= N_per_chirp
    real_if_signal = real_if_signal(1:min(length(real_if_signal), N_per_chirp));
    if length(real_if_signal) < N_per_chirp
        real_if_signal = [real_if_signal; zeros(N_per_chirp - length(real_if_signal), 1)];
    end
end

fprintf('数据预处理完成\n');
fprintf('  中频信号长度: %d\n', length(real_if_signal));

% 1.7 显示真实数据的基本信息
figure('Position', [100, 100, 800, 400], 'Name', '真实雷达数据分析');

subplot(1, 2, 1);
t_us = t_chirp * 1e6;
plot(t_us, real(real_if_signal), 'b-', 'LineWidth', 1.5);
hold on;
plot(t_us, imag(real_if_signal), 'r-', 'LineWidth', 1.5);
xlabel('时间 (μs)');
ylabel('幅度');
title('真实中频信号 (平均后)');
legend('实部', '虚部', 'Location', 'best');
grid on;

subplot(1, 2, 2);
% 计算频谱
window = hamming(N_per_chirp);
real_if_windowed = real_if_signal .* window;
spectrum_real = fft(real_if_windowed, N_per_chirp);
f_axis = (0:N_per_chirp/2-1) * fs / N_per_chirp;  % 频率轴 (Hz)
R_axis = f_axis * c / (2 * slope); % 距离轴 (m)

plot(R_axis*1000, 20*log10(abs(spectrum_real(1:N_per_chirp/2))), 'b-', 'LineWidth', 1.5);
xlabel('距离 (mm)');
ylabel('幅度 (dB)');
title('真实数据距离谱');
grid on;

% 动态设置峰值检测阈值
spectrum_dB = 20*log10(abs(spectrum_real(1:N_per_chirp/2)));
max_dB = max(spectrum_dB);
min_dB = min(spectrum_dB);

if max_dB - min_dB > 20
    min_peak_height = max_dB - 20;
else
    min_peak_height = max_dB - 10;
end

[peaks, locs] = findpeaks(spectrum_dB, 'MinPeakHeight', min_peak_height, ...
    'MinPeakDistance', N_per_chirp/50);

if ~isempty(locs)
    hold on;
    plot(R_axis(locs)*1000, peaks, 'ro', 'MarkerSize', 8, 'MarkerFaceColor', 'r');
    
    % 显示峰值信息
    for i = 1:min(3, length(locs))
        text(R_axis(locs(i))*1000, peaks(i)+2, ...
            sprintf('%.1f mm', R_axis(locs(i))*1000), ...
            'HorizontalAlignment', 'center', 'FontSize', 9);
    end
    
    % 计算可能的厚度（如果检测到两个峰值）
    if length(locs) >= 2
        [sorted_R, idx] = sort(R_axis(locs));
        apparent_thickness = (sorted_R(2) - sorted_R(1)) * 1000;
        fprintf('检测到多个反射面，视在厚度: %.1f mm\n', apparent_thickness);
    end
else
    fprintf('警告：未检测到明显的峰值\n');
end

% 标记距离范围（来自配置文件rangeSelCfg: 0.1 10.0）
hold on;
plot([100, 100], ylim, 'g--', 'LineWidth', 1);  % 0.1 m = 100 mm
plot([10000, 10000], ylim, 'g--', 'LineWidth', 1);  % 10.0 m = 10000 mm
text(100, max_dB-5, '0.1m', 'FontSize', 8, 'Color', 'g');
text(10000, max_dB-5, '10m', 'FontSize', 8, 'Color', 'g');

sgtitle(sprintf('TI IWR6843雷达数据分析 (配置: %d采样点, %.1fMsps)', N_per_chirp, fs/1e6), ...
    'FontSize', 14, 'FontWeight', 'bold');

%% 第二部分：参数估计与模拟数据生成
% =========================================================================
fprintf('\n=== 第二部分：参数估计与模拟数据生成 ===\n');

% 2.1 从真实数据估计前表面距离R0
fprintf('从真实数据估计前表面距离...\n');

if exist('locs', 'var') && ~isempty(locs)
    % 使用第一个峰值（最近的反射面）
    [sorted_R, sort_idx] = sort(R_axis(locs));
    R0_est = sorted_R(1);
    fprintf('使用第一个峰值估计前表面距离: %.6f m (%.2f mm)\n', ...
        R0_est, R0_est*1000);
else
    % 如果没有检测到峰值，寻找频谱最大值的距离
    % 忽略前几个点（可能包含直流分量）
    ignore_points = 10;
    valid_range = R_axis > 0.1 & R_axis < 10.0;  % 使用配置文件的距离范围
    valid_indices = find(valid_range);
    
    if ~isempty(valid_indices)
        valid_indices = valid_indices(valid_indices > ignore_points);
        [~, max_idx] = max(abs(spectrum_real(valid_indices)));
        max_idx = valid_indices(max_idx);
        R0_est = R_axis(max_idx);
        fprintf('使用频谱最大值估计前表面距离: %.6f m (%.2f mm)\n', ...
            R0_est, R0_est*1000);
    else
        % 使用默认值
        R0_est = 0.5;  % 默认0.5米
        fprintf('未检测到有效峰值，使用默认前表面距离: %.6f m (%.2f mm)\n', ...
            R0_est, R0_est*1000);
    end
end

% 2.2 材料参数设置
fprintf('\n材料参数设置...\n');

% 材料选项
material_options = {
    'HDPE (高密度聚乙烯)', 2.35;
    'PVC (聚氯乙烯)', 3.0;
    'PC (聚碳酸酯)', 2.9;
    'PMMA (有机玻璃)', 2.7;
    '木材 (松木)', 2.2;
    '陶瓷', 4.5;
    '未知材料 - 遍历搜索', 0;
};

fprintf('可选材料:\n');
for i = 1:length(material_options)
    if material_options{i, 2} ~= 0
        fprintf('  %d. %s (εr=%.2f)\n', i, material_options{i, 1}, material_options{i, 2});
    else
        fprintf('  %d. %s\n', i, material_options{i, 1});
    end
end

% 选择材料（这里使用默认值HDPE）
material_choice = 1;  % 默认选择HDPE
eps_r_known = material_options{material_choice, 2};

if eps_r_known == 0
    % 使用遍历搜索介电常数
    eps_r_range = [2.0, 3.0, 0.05];  % 最小值, 最大值, 步长（更细的步长）
    fprintf('将遍历介电常数范围: %.2f 到 %.2f, 步长 %.2f\n', ...
        eps_r_range(1), eps_r_range(2), eps_r_range(3));
else
    fprintf('使用已知介电常数: %.2f (%s)\n', ...
        eps_r_known, material_options{material_choice, 1});
end

% 2.3 厚度搜索范围设置
fprintf('\n厚度搜索范围设置...\n');

% 根据配置文件的距离范围设置厚度范围
d_min = 0.07;   % 
d_max = 0.09;   % 

% 如果检测到多个峰值，可以根据峰值间距调整范围
if exist('locs', 'var') && length(locs) >= 2
    [sorted_R, ~] = sort(R_axis(locs));
    apparent_thickness = sorted_R(2) - sorted_R(1);
    
    % 假设折射率约为1.5，估算实际厚度
    estimated_thickness = apparent_thickness / 1.5;
    
    % 以估计厚度为中心调整搜索范围
    d_min = max(d_min, estimated_thickness * 0.3);
    d_max = min(d_max, estimated_thickness * 3.0);
    
    fprintf('根据峰值间距调整厚度范围:\n');
    fprintf('  视在厚度: %.2f mm\n', apparent_thickness*1000);
    fprintf('  估计厚度: %.2f mm\n', estimated_thickness*1000);
end

% 厚度搜索步长设置 - 使用更精细的步长
d_step_fine = 0.0001;  % 0.1 mm步长（更精细）

% 显示厚度搜索范围
fprintf('厚度搜索范围:\n');
fprintf('  最小值: %.2f mm\n', d_min*1000);
fprintf('  最大值: %.2f mm\n', d_max*1000);
fprintf('  精细搜索步长: %.3f mm\n', d_step_fine*1000);

% 2.4 生成发射信号（用于模拟数据生成）
fprintf('\n生成发射信号用于模拟...\n');
Tx_signal = exp(1i * 2 * pi * (f0 * t_chirp + 0.5 * slope * t_chirp.^2));

%% 第三部分：模拟数据生成与对比 - 改进版
% =========================================================================
fprintf('\n=== 第三部分：模拟数据生成与对比（改进版） ===\n');

% 3.1 智能搜索策略：先粗后细，逐步细化
num_refinement_stages = 3;  % 细化阶段数
d_step_stages = [0.001, 0.0002, 0.00005];  % 每个阶段的步长 (m) [1mm, 0.2mm, 0.05mm]
refinement_factor = 0.2;    % 每个阶段搜索范围缩小比例

% 初始化最佳结果
best_d_est = (d_min + d_max) / 2;  % 初始猜测为中间值
best_eps_r = eps_r_known;
if eps_r_known == 0
    best_eps_r = 2.5;  % 初始猜测
end
best_correlation = 0;
best_sim_signal = [];

% 对介电常数的处理
if eps_r_known == 0
    % 需要遍历介电常数
    eps_r_test_values = eps_r_range(1):eps_r_range(3):eps_r_range(2);
    num_eps_r_tests = length(eps_r_test_values);
    fprintf('介电常数测试点数: %d\n', num_eps_r_tests);
else
    eps_r_test_values = eps_r_known;
    num_eps_r_tests = 1;
end

% 3.4 多阶段细化搜索
fprintf('开始多阶段细化搜索...\n');
total_tests = 0;
total_multi_stage_time = 0;  % 初始化总时间

for stage = 1:num_refinement_stages
    fprintf('\n--- 第%d阶段搜索 ---\n', stage);
    
    % 当前阶段的步长
    current_d_step = d_step_stages(stage);
    
    % 确定当前阶段的搜索范围
    if stage == 1
        % 第一阶段：全局搜索
        d_current_min = d_min;
        d_current_max = d_max;
    else
        % 后续阶段：基于上一阶段最佳结果缩小范围
        search_range = (d_max - d_min) * refinement_factor;
        d_current_min = max(d_min, best_d_est - search_range);
        d_current_max = min(d_max, best_d_est + search_range);
    end
    
    % 创建搜索向量
    d_test_values = d_current_min:current_d_step:d_current_max;
    num_d_tests = length(d_test_values);
    
    fprintf('  厚度搜索范围: %.2f 到 %.2f mm\n', d_current_min*1000, d_current_max*1000);
    fprintf('  厚度步长: %.3f mm\n', current_d_step*1000);
    fprintf('  厚度测试点数: %d\n', num_d_tests);
    fprintf('  介电常数测试点数: %d\n', num_eps_r_tests);
    
    % 初始化结果存储
    if num_eps_r_tests > 1
        correlation_matrix = zeros(num_d_tests, num_eps_r_tests);
    else
        correlation_vector = zeros(num_d_tests, 1);
    end
    
    % 执行搜索
    stage_start_time = tic;
    stage_tests = 0;
    
    for i = 1:num_d_tests
        d_test = d_test_values(i);
        
        for j = 1:num_eps_r_tests
            if eps_r_known == 0
                eps_r_test = eps_r_test_values(j);
            else
                eps_r_test = eps_r_known;
            end
            
            % 生成模拟信号
            sim_if_signal = generate_sim_signal(t_chirp, f0, slope, R0_est, d_test, eps_r_test, Tx_signal, c);
            
            % 计算相关系数
            correlation = compute_correlation(real_if_signal, sim_if_signal);
            
            % 存储结果
            if num_eps_r_tests > 1
                correlation_matrix(i, j) = correlation;
            else
                correlation_vector(i) = correlation;
            end
            
            % 更新全局最佳匹配
            if correlation > best_correlation
                best_correlation = correlation;
                best_d_est = d_test;
                if eps_r_known == 0
                    best_eps_r = eps_r_test;
                end
                best_sim_signal = sim_if_signal;
            end
            
            stage_tests = stage_tests + 1;
            total_tests = total_tests + 1;
        end
        
        % 显示进度
        if mod(i, max(1, floor(num_d_tests/10))) == 0
            progress = i / num_d_tests * 100;
            fprintf('    进度: %.1f%% (厚度 %.2f/%.2f mm)\n', ...
                progress, d_test*1000, d_current_max*1000);
        end
    end
    
    stage_elapsed_time = toc(stage_start_time);
    total_multi_stage_time = total_multi_stage_time + stage_elapsed_time;  % 累加时间
    
    fprintf('  第%d阶段完成! 耗时: %.2f 秒\n', stage, stage_elapsed_time);
    fprintf('  当前最佳结果: 厚度=%.4f mm, 相关系数=%.6f\n', ...
        best_d_est*1000, best_correlation);
    
    % 可视化当前阶段结果
    if stage < num_refinement_stages
        figure('Position', [100, 100, 800, 400], 'Name', sprintf('第%d阶段搜索结果', stage));
        
        subplot(1, 2, 1);
        plot(d_test_values*1000, correlation_vector, 'b-', 'LineWidth', 2);
        hold on;
        plot(best_d_est*1000, best_correlation, 'ro', 'MarkerSize', 10, 'MarkerFaceColor', 'r');
        xlabel('厚度 (mm)');
        ylabel('相关系数');
        title(sprintf('第%d阶段: 相关系数随厚度变化', stage));
        grid on;
        
        subplot(1, 2, 2);
        % 绘制最佳匹配信号对比
        t_us = t_chirp * 1e6;
        plot(t_us, real(real_if_signal), 'b-', 'LineWidth', 1.5);
        hold on;
        plot(t_us, real(best_sim_signal), 'r--', 'LineWidth', 1.5);
        xlabel('时间 (μs)');
        ylabel('幅度 (实部)');
        title(sprintf('第%d阶段: 最佳匹配信号', stage));
        legend('真实数据', '模拟数据', 'Location', 'best');
        grid on;
        
        sgtitle(sprintf('第%d阶段搜索结果 - 最佳厚度: %.2f mm', stage, best_d_est*1000), ...
            'FontSize', 12, 'FontWeight', 'bold');
    end
end

% 3.5 最终细化搜索（使用更精细的步长和插值）
fprintf('\n=== 最终细化搜索 ===\n');

% 使用非常精细的步长
d_step_final = 0.00001;  % 0.01 mm步长
final_range = 0.001;     % ±1 mm范围

d_final_min = max(d_min, best_d_est - final_range);
d_final_max = min(d_max, best_d_est + final_range);
d_final_values = d_final_min:d_step_final:d_final_max;
num_final_tests = length(d_final_values);

fprintf('最终细化搜索参数:\n');
fprintf('  搜索范围: %.3f 到 %.3f mm\n', d_final_min*1000, d_final_max*1000);
fprintf('  搜索步长: %.4f mm\n', d_step_final*1000);
fprintf('  测试点数: %d\n', num_final_tests);

final_correlations = zeros(num_final_tests, 1);
best_final_correlation = 0;
best_final_d = best_d_est;

final_start_time = tic;

for i = 1:num_final_tests
    d_test_final = d_final_values(i);
    
    % 生成模拟信号
    sim_if_signal_final = generate_sim_signal(t_chirp, f0, slope, R0_est, d_test_final, best_eps_r, Tx_signal, c);
    
    % 计算相关系数
    correlation_final = compute_correlation(real_if_signal, sim_if_signal_final);
    
    final_correlations(i) = correlation_final;
    
    % 更新最佳值
    if correlation_final > best_final_correlation
        best_final_correlation = correlation_final;
        best_final_d = d_test_final;
        best_sim_signal = sim_if_signal_final;
    end
end

final_elapsed_time = toc(final_start_time);
fprintf('最终细化搜索完成! 耗时: %.2f 秒\n', final_elapsed_time);

% 3.6 使用抛物线插值进一步提高精度
if num_final_tests >= 3
    fprintf('\n使用抛物线插值进一步提高精度...\n');
    
    % 找到最佳点及其相邻点
    [~, best_idx] = max(final_correlations);
    
    if best_idx > 1 && best_idx < num_final_tests
        % 提取三个点进行抛物线插值
        x1 = d_final_values(best_idx-1);
        x2 = d_final_values(best_idx);
        x3 = d_final_values(best_idx+1);
        
        y1 = final_correlations(best_idx-1);
        y2 = final_correlations(best_idx);
        y3 = final_correlations(best_idx+1);
        
        % 抛物线插值公式
        denom = (x1-x2)*(x1-x3)*(x2-x3);
        A = (x3*(y2-y1) + x2*(y1-y3) + x1*(y3-y2)) / denom;
        B = (x3^2*(y1-y2) + x2^2*(y3-y1) + x1^2*(y2-y3)) / denom;
        C = (x2*x3*(x2-x3)*y1 + x3*x1*(x3-x1)*y2 + x1*x2*(x1-x2)*y3) / denom;
        
        % 抛物线顶点
        if A < 0  % 确保是最大值
            x_peak = -B/(2*A);
            y_peak = C - B^2/(4*A);
            
            % 检查插值点是否在范围内
            if x_peak >= d_final_min && x_peak <= d_final_max
                best_final_d = x_peak;
                best_final_correlation = y_peak;
                
                % 重新生成最佳模拟信号
                best_sim_signal = generate_sim_signal(t_chirp, f0, slope, R0_est, best_final_d, best_eps_r, Tx_signal, c);
                
                fprintf('抛物线插值结果: 厚度=%.6f mm, 相关系数=%.8f\n', ...
                    best_final_d*1000, best_final_correlation);
            end
        end
    end
end

% 3.7 更新最终结果
best_d_est = best_final_d;
best_correlation = best_final_correlation;

% 计算总时间
total_computation_time = total_multi_stage_time + final_elapsed_time;

% 计算理论距离分辨率
range_resolution = c / (2 * BW);

fprintf('\n最终最佳匹配结果:\n');
fprintf('  估计厚度: %.6f m (%.4f mm)\n', best_d_est, best_d_est*1000);
fprintf('  使用介电常数: %.4f (折射率: %.4f)\n', best_eps_r, sqrt(best_eps_r));
fprintf('  相关系数: %.8f\n', best_correlation);
fprintf('  理论距离分辨率: %.3f mm\n', range_resolution*1000);
fprintf('  总测试点数: %d\n', total_tests + num_final_tests);
fprintf('  总计算时间: %.2f 秒\n', total_computation_time);

%% 第四部分：结果分析与可视化
% =========================================================================
fprintf('\n=== 第四部分：结果分析与可视化 ===\n');

% 4.1 创建主结果图
figure('Position', [50, 50, 1400, 900], 'Name', 'TI IWR6843雷达厚度测量结果（改进版）');

% 子图1: 真实数据与最佳匹配模拟数据对比
subplot(3, 4, 1);
t_us = t_chirp * 1e6;
plot(t_us, real(real_if_signal), 'b-', 'LineWidth', 2);
hold on;
plot(t_us, real(best_sim_signal), 'r--', 'LineWidth', 2);
xlabel('时间 (μs)');
ylabel('幅度 (实部)');
title('中频信号对比 (实部)');
legend('真实数据', sprintf('模拟数据 (%.3fmm)', best_d_est*1000), 'Location', 'best');
grid on;

% 添加相关系数显示
text(0.05, 0.9, sprintf('相关系数: %.8f', best_correlation), ...
    'Units', 'normalized', 'FontSize', 10, 'Color', 'k');

% 子图2: 距离谱对比
subplot(3, 4, 2);
% 计算频谱
window = hamming(N_per_chirp);
real_spectrum = fft(real_if_signal .* window, N_per_chirp);
sim_spectrum = fft(best_sim_signal .* window, N_per_chirp);

plot(R_axis*1000, 20*log10(abs(real_spectrum(1:N_per_chirp/2))), 'b-', 'LineWidth', 2);
hold on;
plot(R_axis*1000, 20*log10(abs(sim_spectrum(1:N_per_chirp/2))), 'r--', 'LineWidth', 2);
xlabel('距离 (mm)');
ylabel('幅度 (dB)');
title('距离谱对比');
grid on;

% 标记配置的距离范围
plot([100, 100], ylim, 'g--', 'LineWidth', 1);
plot([10000, 10000], ylim, 'g--', 'LineWidth', 1);
text(500, max(20*log10(abs(real_spectrum(1:N_per_chirp/2))))-10, '配置范围: 0.1-10m', ...
    'FontSize', 9, 'Color', 'g');

legend('真实数据', '模拟数据', '配置范围', 'Location', 'best');

% 子图3: 最终细化搜索结果
subplot(3, 4, 3);
plot(d_final_values*1000, final_correlations, 'b-', 'LineWidth', 2);
hold on;
plot([best_d_est*1000, best_d_est*1000], [min(final_correlations), max(final_correlations)], ...
    'r--', 'LineWidth', 1.5);
plot(best_d_est*1000, best_correlation, 'ro', 'MarkerSize', 10, 'MarkerFaceColor', 'r');
xlabel('厚度 (mm)');
ylabel('相关系数');
title('最终细化搜索结果');
grid on;

% 添加标记
text(best_d_est*1000, best_correlation, sprintf('最佳: %.3fmm', best_d_est*1000), ...
    'VerticalAlignment', 'bottom', 'HorizontalAlignment', 'center', ...
    'FontSize', 9, 'Color', 'r');

% 显示搜索范围
xlim([d_final_min*1000, d_final_max*1000]);

% 子图4: 系统参数显示
subplot(3, 4, 4);
text(0.1, 0.9, '雷达系统参数', 'FontSize', 12, 'FontWeight', 'bold');
text(0.1, 0.85, sprintf('起始频率: %.2f GHz', f0/1e9), 'FontSize', 10);
text(0.1, 0.80, sprintf('斜率: %.0f MHz/μs', slope_MHz_per_us), 'FontSize', 10);
text(0.1, 0.75, sprintf('带宽: %.3f GHz', BW/1e9), 'FontSize', 10);
text(0.1, 0.70, sprintf('采样率: %.2f Msps', fs/1e6), 'FontSize', 10);
text(0.1, 0.65, sprintf('采样点数: %d', N_per_chirp), 'FontSize', 10);
text(0.1, 0.60, sprintf('啁啾时间: %.1f μs', Tc*1e6), 'FontSize', 10);
text(0.1, 0.55, sprintf('距离分辨率: %.3f mm', range_resolution*1000), 'FontSize', 10);
text(0.1, 0.50, sprintf('通道: %dTX × %dRX', n_tx_chan, n_rx_chan), 'FontSize', 10);
text(0.1, 0.45, sprintf('每帧啁啾数: %d', n_chirps_per_frame), 'FontSize', 10);
text(0.1, 0.40, sprintf('总帧数: %d', n_frames), 'FontSize', 10);

axis off;

% 子图5: 信号幅度对比
subplot(3, 4, 5);
plot(t_us, abs(real_if_signal), 'b-', 'LineWidth', 2);
hold on;
plot(t_us, abs(best_sim_signal), 'r--', 'LineWidth', 2);
xlabel('时间 (μs)');
ylabel('幅度');
title('信号幅度对比');
legend('真实数据', '模拟数据', 'Location', 'best');
grid on;

% 子图6: 信号相位对比
subplot(3, 4, 6);
phase_real = unwrap(angle(real_if_signal));
phase_sim = unwrap(angle(best_sim_signal));

plot(t_us, phase_real, 'b-', 'LineWidth', 2);
hold on;
plot(t_us, phase_sim, 'r--', 'LineWidth', 2);
xlabel('时间 (μs)');
ylabel('相位 (rad)');
title('信号相位对比');
legend('真实数据', '模拟数据', 'Location', 'best');
grid on;

% 子图7: 信号残差
subplot(3, 4, 7);
residual = real_if_signal - best_sim_signal;
plot(t_us, real(residual), 'b-', 'LineWidth', 1.5);
hold on;
plot(t_us, imag(residual), 'r-', 'LineWidth', 1.5);
xlabel('时间 (μs)');
ylabel('残差幅度');
title('信号残差');
legend('实部残差', '虚部残差', 'Location', 'best');
grid on;

% 计算残差统计
residual_power = mean(abs(residual).^2);
signal_power = mean(abs(real_if_signal).^2);
snr_est = 10*log10(signal_power / residual_power);

text(0.05, 0.9, sprintf('残差功率: %.2e', residual_power), ...
    'Units', 'normalized', 'FontSize', 9);
text(0.05, 0.85, sprintf('信号功率: %.2e', signal_power), ...
    'Units', 'normalized', 'FontSize', 9);
text(0.05, 0.80, sprintf('估计SNR: %.1f dB', snr_est), ...
    'Units', 'normalized', 'FontSize', 9);

% 子图8: 厚度估计结果
subplot(3, 4, 8);
bar_values = [best_d_est*1000];
bar_labels = {'估计厚度'};

bar(1, bar_values, 'FaceColor', [0.3, 0.6, 0.9]);
ylabel('厚度 (mm)');
title('厚度估计结果');
grid on;

% 添加数值标签
text(1, bar_values+max(bar_values)*0.1, sprintf('%.3f mm', bar_values), ...
    'HorizontalAlignment', 'center', 'FontSize', 12, 'FontWeight', 'bold');

% 显示估计精度
text(1, bar_values*0.5, sprintf('精度: ±%.4f mm', d_step_final*1000/2), ...
    'HorizontalAlignment', 'center', 'FontSize', 10);

set(gca, 'XTickLabel', bar_labels);
ylim([0, best_d_est*1000*1.5]);

% 子图9: 传播路径示意图
subplot(3, 4, 9);
% 绘制几何示意图
x = [0, 1, 1, 0];
y = [0, 0, 0.8, 0.8];
fill(x, y, [0.9, 0.9, 0.9], 'EdgeColor', 'k', 'LineWidth', 2);
hold on;

% 雷达位置
plot(0.5, 1.2, '^', 'MarkerSize', 20, 'MarkerFaceColor', 'b', 'Color', 'b');
text(0.5, 1.3, '雷达', 'HorizontalAlignment', 'center', 'FontSize', 10);

% 传播路径
% 信号A路径（前表面反射）
plot([0.5, 0.5], [1.2, 0.8], 'r-', 'LineWidth', 2);
plot([0.5, 0.5], [0.8, 1.2], 'r-', 'LineWidth', 2);
text(0.6, 1.0, '信号A', 'Color', 'r', 'FontSize', 10);

% 信号B路径（后表面反射）
plot([0.5, 0.5], [1.2, 0], 'g-', 'LineWidth', 2);
plot([0.5, 0.5], [0, 1.2], 'g-', 'LineWidth', 2);
text(0.6, 0.6, '信号B', 'Color', 'g', 'FontSize', 10);

% 标记距离
text(0.5, 0.75, sprintf('R₀=%.2f mm', R0_est*1000), ...
    'HorizontalAlignment', 'center', 'FontSize', 9, 'Interpreter', 'none');
text(0.5, 0.05, sprintf('R₀+d=%.2f mm', (R0_est+best_d_est*sqrt(best_eps_r))*1000), ...
    'HorizontalAlignment', 'center', 'FontSize', 9, 'Interpreter', 'none');
text(0.5, 0.4, sprintf('厚度=%.3f mm', best_d_est*1000), ...
    'HorizontalAlignment', 'center', 'FontSize', 10, 'FontWeight', 'bold', 'Interpreter', 'none');

axis equal;
axis([0, 1, -0.1, 1.4]);
set(gca, 'XTick', [], 'YTick', []);
title('信号传播路径');
box on;

% 子图10: 材料参数
subplot(3, 4, 10);
text(0.1, 0.9, '材料参数', 'FontSize', 12, 'FontWeight', 'bold');
text(0.1, 0.85, sprintf('材料: %s', material_options{material_choice, 1}), 'FontSize', 10);
text(0.1, 0.80, sprintf('介电常数: %.4f', best_eps_r), 'FontSize', 10);
text(0.1, 0.75, sprintf('折射率: %.4f', sqrt(best_eps_r)), 'FontSize', 10);

% 计算反射系数
n_material = sqrt(best_eps_r);
Gamma1 = (1 - n_material) / (1 + n_material);
Gamma2 = (n_material - 1) / (n_material + 1);
text(0.1, 0.70, sprintf('前表面反射系数: %.4f', Gamma1), 'FontSize', 10);
text(0.1, 0.65, sprintf('后表面反射系数: %.4f', Gamma2), 'FontSize', 10);

axis off;

% 子图11: 搜索算法信息
subplot(3, 4, 11);
text(0.1, 0.9, '搜索算法', 'FontSize', 12, 'FontWeight', 'bold');
text(0.1, 0.85, '多阶段细化搜索:', 'FontSize', 10);
text(0.1, 0.80, sprintf('  阶段1: 步长 %.2f mm', d_step_stages(1)*1000), 'FontSize', 9);
text(0.1, 0.75, sprintf('  阶段2: 步长 %.2f mm', d_step_stages(2)*1000), 'FontSize', 9);
text(0.1, 0.70, sprintf('  阶段3: 步长 %.2f mm', d_step_stages(3)*1000), 'FontSize', 9);
text(0.1, 0.65, sprintf('  最终细化: 步长 %.3f mm', d_step_final*1000), 'FontSize', 9);

text(0.1, 0.55, '优化技术:', 'FontSize', 10);
text(0.1, 0.50, '  抛物线插值', 'FontSize', 9);
text(0.1, 0.45, sprintf('  总测试点数: %d', total_tests + num_final_tests), 'FontSize', 9);
text(0.1, 0.40, sprintf('  总计算时间: %.2f 秒', total_computation_time), 'FontSize', 9);

axis off;

% 子图12: 结果总结
subplot(3, 4, 12);
text(0.1, 0.9, '测量结果总结', 'FontSize', 12, 'FontWeight', 'bold');
text(0.1, 0.85, sprintf('厚度估计:'), 'FontSize', 10);
text(0.1, 0.80, sprintf('  %.3f mm', best_d_est*1000), 'FontSize', 12, 'FontWeight', 'bold');

text(0.1, 0.75, sprintf('前表面距离:'), 'FontSize', 10);
text(0.1, 0.70, sprintf('  %.2f mm', R0_est*1000), 'FontSize', 10);

text(0.1, 0.65, sprintf('匹配质量:'), 'FontSize', 10);
text(0.1, 0.60, sprintf('  相关系数: %.8f', best_correlation), 'FontSize', 10);

text(0.1, 0.55, sprintf('理论分辨率:'), 'FontSize', 10);
text(0.1, 0.50, sprintf('  %.3f mm', range_resolution*1000), 'FontSize', 10);

text(0.1, 0.45, sprintf('估计精度:'), 'FontSize', 10);
text(0.1, 0.40, sprintf('  ±%.4f mm', d_step_final*1000/2), 'FontSize', 10);

axis off;

% 总标题
sgtitle(sprintf('TI IWR6843雷达厚度测量 - 估计厚度: %.3f mm', best_d_est*1000), ...
    'FontSize', 14, 'FontWeight', 'bold');

%% 第五部分：误差分析与不确定性评估
% =========================================================================
fprintf('\n=== 第五部分：误差分析与不确定性评估 ===\n');

% 5.1 计算不确定性区间
% 基于相关系数曲线确定半高全宽(FWHM)作为不确定性度量
fprintf('计算厚度估计的不确定性...\n');

% 寻找相关系数下降一半的点
half_max = best_correlation - (best_correlation - min(final_correlations)) * 0.5;

% 寻找交叉点
above_half = final_correlations >= half_max;
cross_points = d_final_values(above_half) * 1000;

if ~isempty(cross_points)
    uncertainty_range = (max(cross_points) - min(cross_points)) / 2;
    fprintf('  基于半高全宽的不确定性: ±%.4f mm\n', uncertainty_range);
else
    % 使用最终步长作为不确定性估计
    uncertainty_range = d_step_final * 1000 / 2;
    fprintf('  使用步长估计的不确定性: ±%.4f mm\n', uncertainty_range);
end

% 5.2 计算信噪比相关的误差
fprintf('计算信噪比相关的误差...\n');

% 估计噪声水平（使用信号尾部的平均值）
noise_region_start = floor(N_per_chirp * 0.8);  % 最后20%作为噪声估计
noise_power = mean(abs(real_if_signal(noise_region_start:end)).^2);
signal_power_noise_est = mean(abs(real_if_signal(1:noise_region_start)).^2);
snr_db = 10*log10(signal_power_noise_est / noise_power);

% 基于SNR的距离不确定性
range_uncertainty_snr = range_resolution / sqrt(snr_db/10);  % 近似公式
fprintf('  估计SNR: %.1f dB\n', snr_db);
fprintf('  基于SNR的距离不确定性: %.4f mm\n', range_uncertainty_snr*1000);

% 5.3 系统误差分析
fprintf('系统误差分析...\n');

% 距离分辨率限制的误差
range_resolution_error = range_resolution / 2;  % 理论极限

% 采样率相关的误差
sampling_error = c / (2 * fs * N_per_chirp);  % 采样时间决定的精度

% 综合误差估计
total_uncertainty = sqrt(uncertainty_range^2 + (range_uncertainty_snr*1000)^2 + ...
    (range_resolution_error*1000)^2 + (sampling_error*1000)^2);

fprintf('  距离分辨率限制: ±%.4f mm\n', range_resolution_error*1000);
fprintf('  采样率限制: ±%.4f mm\n', sampling_error*1000);
fprintf('  综合不确定性: ±%.4f mm (95%%置信区间)\n', total_uncertainty);

%% 第六部分：结果保存与报告生成
% =========================================================================
fprintf('\n=== 第六部分：结果保存 ===\n');

% 6.1 创建结果结构体
results = struct();

% 系统参数
results.system_params.c = c;
results.system_params.f0 = f0;
results.system_params.slope_MHz_per_us = slope_MHz_per_us;
results.system_params.slope = slope;
results.system_params.fs = fs;
results.system_params.N_per_chirp = N_per_chirp;
results.system_params.Tc = Tc;
results.system_params.actual_sampling_time = actual_sampling_time;
results.system_params.BW = BW;
results.system_params.range_resolution = range_resolution;

% 系统配置
results.system_config.n_rx_chan = n_rx_chan;
results.system_config.n_tx_chan = n_tx_chan;
results.system_config.virtual_channels = virtual_channels;
results.system_config.n_chirps_per_frame = n_chirps_per_frame;
results.system_config.n_frames = n_frames;

% 测量参数
results.measurement_params.R0_est = R0_est;
results.measurement_params.d_min = d_min;
results.measurement_params.d_max = d_max;
results.measurement_params.d_step_final = d_step_final;
results.measurement_params.eps_r_used = best_eps_r;
results.measurement_params.material = material_options{material_choice, 1};

% 测量结果
results.measurement_results.d_est = best_d_est;
results.measurement_results.correlation = best_correlation;
results.measurement_results.R0_est = R0_est;
results.measurement_results.range_resolution = range_resolution;

% 不确定性分析
results.uncertainty_analysis.uncertainty_range = uncertainty_range;
results.uncertainty_analysis.snr_db = snr_db;
results.uncertainty_analysis.range_uncertainty_snr = range_uncertainty_snr;
results.uncertainty_analysis.range_resolution_error = range_resolution_error;
results.uncertainty_analysis.sampling_error = sampling_error;
results.uncertainty_analysis.total_uncertainty = total_uncertainty;

% 计算信息
results.computation_info.total_tests = total_tests + num_final_tests;
results.computation_info.final_tests = num_final_tests;
results.computation_info.total_time = total_computation_time;
results.computation_info.timestamp = datestr(now);
results.computation_info.data_file = real_data_file;
results.computation_info.algorithm = '多阶段细化搜索 + 抛物线插值';

% 6.2 保存结果到MAT文件
results_file = 'ti_iwr6843_thickness_measurement_results_improved.mat';
save(results_file, 'results', 'real_if_signal', 'best_sim_signal', 'd_final_values', 'final_correlations');
fprintf('结果已保存到: %s\n', results_file);

% 6.3 生成详细的文本报告
report_file = 'ti_iwr6843_thickness_measurement_report_improved.txt';
fid = fopen(report_file, 'w');
if fid == -1
    fprintf('警告: 无法创建报告文件\n');
else
    fprintf(fid, 'TI IWR6843雷达厚度测量报告（改进版）\n');
    fprintf(fid, '生成时间: %s\n\n', datestr(now));
    
    fprintf(fid, '=== 系统参数 (基于配置文件) ===\n');
    fprintf(fid, '起始频率: %.2f GHz\n', f0/1e9);
    fprintf(fid, '斜率: %.0f MHz/μs\n', slope_MHz_per_us);
    fprintf(fid, '采样率: %.2f Msps\n', fs/1e6);
    fprintf(fid, '每啁啾采样点数: %d\n', N_per_chirp);
    fprintf(fid, '啁啾斜坡时间: %.1f μs\n', Tc*1e6);
    fprintf(fid, '实际采样时间: %.2f μs\n', actual_sampling_time*1e6);
    fprintf(fid, '带宽: %.3f GHz\n', BW/1e9);
    fprintf(fid, '距离分辨率: %.3f mm\n\n', range_resolution*1000);
    
    fprintf(fid, '=== 系统配置 ===\n');
    fprintf(fid, '接收通道数: %d\n', n_rx_chan);
    fprintf(fid, '发射通道数: %d\n', n_tx_chan);
    fprintf(fid, '虚拟通道数: %d\n', virtual_channels);
    fprintf(fid, '每帧啁啾数: %d\n', n_chirps_per_frame);
    fprintf(fid, '总帧数: %d\n\n', n_frames);
    
    fprintf(fid, '=== 测量参数 ===\n');
    fprintf(fid, '前表面距离估计: %.6f m (%.2f mm)\n', R0_est, R0_est*1000);
    fprintf(fid, '厚度搜索范围: %.2f - %.2f mm\n', d_min*1000, d_max*1000);
    fprintf(fid, '最终搜索步长: %.4f mm\n', d_step_final*1000);
    fprintf(fid, '使用材料: %s\n', material_options{material_choice, 1});
    fprintf(fid, '介电常数: %.4f\n', best_eps_r);
    fprintf(fid, '折射率: %.4f\n\n', sqrt(best_eps_r));
    
    fprintf(fid, '=== 测量结果 ===\n');
    fprintf(fid, '厚度估计: %.6f m (%.4f mm\n', best_d_est, best_d_est*1000);
    fprintf(fid, '相关系数: %.8f\n', best_correlation);
    fprintf(fid, '前表面距离: %.6f m (%.2f mm)\n\n', R0_est, R0_est*1000);
    
    fprintf(fid, '=== 不确定性分析 ===\n');
    fprintf(fid, '基于半高全宽的不确定性: ±%.4f mm\n', uncertainty_range);
    fprintf(fid, '估计SNR: %.1f dB\n', snr_db);
    fprintf(fid, '基于SNR的距离不确定性: ±%.4f mm\n', range_uncertainty_snr*1000);
    fprintf(fid, '距离分辨率限制: ±%.4f mm\n', range_resolution_error*1000);
    fprintf(fid, '采样率限制: ±%.4f mm\n', sampling_error*1000);
    fprintf(fid, '综合不确定性: ±%.4f mm (95%%置信区间)\n\n', total_uncertainty);
    
    fprintf(fid, '=== 理论性能 ===\n');
    fprintf(fid, '理论距离分辨率: %.3f mm\n\n', range_resolution*1000);
    
    fprintf(fid, '=== 计算信息 ===\n');
    fprintf(fid, '搜索算法: %s\n', results.computation_info.algorithm);
    fprintf(fid, '总测试点数: %d\n', results.computation_info.total_tests);
    fprintf(fid, '总计算时间: %.2f 秒\n', results.computation_info.total_time);
    fprintf(fid, '数据文件: %s\n', real_data_file);
    
    fclose(fid);
    fprintf('报告已保存到: %s\n', report_file);
end

% 6.4 显示最终总结
fprintf('\n=== 最终结果总结 ===\n');
fprintf('测量完成!\n');
fprintf('最终厚度估计: %.4f mm\n', best_d_est*1000);
fprintf('材料: %s (εr=%.4f)\n', material_options{material_choice, 1}, best_eps_r);
fprintf('前表面距离: %.2f mm\n', R0_est*1000);
fprintf('相关系数: %.8f\n', best_correlation);
fprintf('理论距离分辨率: %.3f mm\n', range_resolution*1000);
fprintf('综合不确定性: ±%.4f mm (95%%置信区间)\n', total_uncertainty);
fprintf('总计算时间: %.2f 秒\n', total_computation_time);
fprintf('结果文件: %s\n', results_file);
fprintf('报告文件: %s\n', report_file);
fprintf('\n程序执行完毕！\n');

%% 函数定义部分
% 放在脚本的最后

% 生成模拟信号的函数
function sim_if = generate_sim_signal(t_chirp, f0, slope, R0_est, d_test, eps_r_test, Tx_signal, c)
    % 计算折射率
    n_test = sqrt(eps_r_test);
    
    % 计算反射系数（菲涅尔公式）
    n_air = 1;
    Gamma1_test = (n_air - n_test) / (n_air + n_test);  % 前表面反射系数
    Gamma2_test = (n_test - n_air) / (n_test + n_air);  % 后表面反射系数
    T12_test = 1 + Gamma1_test;  % 空气到材料的传输系数
    T21_test = 1 + Gamma2_test;  % 材料到空气的传输系数
    
    % 计算延迟
    tau_A_test = 2 * R0_est / c;
    tau_B_test = (2*R0_est)/c + (2*d_test*n_test)/c;
    
    % 生成模拟信号
    echo_A_test = abs(Gamma1_test) * exp(1i * angle(Gamma1_test)) * ...
                  exp(1i * 2 * pi * (f0 * (t_chirp - tau_A_test) + 0.5 * slope * (t_chirp - tau_A_test).^2));
    
    echo_B_test = abs(T12_test * Gamma2_test * T21_test) * ...
                  exp(1i * angle(T12_test * Gamma2_test * T21_test)) * ...
                  exp(1i * 2 * pi * (f0 * (t_chirp - tau_B_test) + 0.5 * slope * (t_chirp - tau_B_test).^2));
    
    Rx_test = echo_A_test + echo_B_test;
    sim_if = Tx_signal .* conj(Rx_test);
end

% 计算相关系数的函数
function corr = compute_correlation(real_signal, sim_signal)
    real_signal_norm = real_signal / norm(real_signal);
    sim_signal_norm = sim_signal / norm(sim_signal);
    corr = abs(dot(real_signal_norm, sim_signal_norm));
end