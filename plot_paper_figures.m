%% plot_paper_figures.m
% 论文图表绘制脚本
% 需要先运行: python export_paper_data.py

clear; clc; close all;

result_dir = 'results';

%% ============================================================
%  图1: 散点图 - 三张独立图
%  ============================================================

data = load(fullfile(result_dir, 'scatter_data.mat'));

cond_names = {'Static', 'Sinusoidal', 'Stochastic'};
file_names = {'scatter_static.pdf', 'scatter_sinusoidal.pdf', 'scatter_stochastic.pdf'};
colors = [0.2 0.6 0.8;    % static  - 蓝
          0.9 0.4 0.1;    % sine    - 橙
          0.3 0.7 0.3];   % noise   - 绿
markers = {'o', 's', '^'};

% 统一坐标轴范围（使三张图可比）
ax_lim = [min(data.y_true_mm(:)) - 0.1, max(data.y_true_mm(:)) + 0.1];

for cid = 1:3
    mask = data.condition_id == cid;

    y_true_sub = data.y_true_mm(mask);
    y_pred_sub = data.y_pred_mm(mask);

    figure('Units', 'centimeters', 'Position', [2 2 9 8]);
    hold on;

    scatter(y_true_sub, y_pred_sub, ...
        4, colors(cid, :), markers{cid}, ...
        'filled', 'MarkerFaceAlpha', 0.3, 'MarkerEdgeAlpha', 0.3);

    % y = x 参考线
    plot(ax_lim, ax_lim, 'k--', 'LineWidth', 1.0);

    xlim(ax_lim); ylim(ax_lim);
    axis square;
    xlabel('Measured h (mm)');
    ylabel('Estimated h (mm)');
    title(cond_names{cid}, 'FontSize', 9);

    % 仅计算当前条件的 R²
    err = y_pred_sub(:) - y_true_sub(:);
    ss_res = sum(err.^2);
    ss_tot = sum((y_true_sub(:) - mean(y_true_sub(:))).^2);
    r2 = 1 - ss_res / ss_tot;
    text(ax_lim(2) - 0.05, ax_lim(1) + 0.15, ...
        sprintf('R^2 = %.4f', r2), ...
        'FontSize', 8, 'HorizontalAlignment', 'right');

    set(gca, 'FontSize', 8, 'TickDir', 'in', 'Box', 'on');
    hold off;

    exportgraphics(gcf, fullfile(result_dir, file_names{cid}), ...
        'ContentType', 'vector');
    fprintf('已保存: %s\n', file_names{cid});
end

%% ============================================================
%  图2: 时序图 (representative_case.pdf)
%  ============================================================

ts = load(fullfile(result_dir, 'timeseries_data.mat'));

figure('Units', 'centimeters', 'Position', [2 2 18 12]);

panels = {'static', 'sine', 'noise'};
titles = {'(a) Static equilibrium', ...
          '(b) Sinusoidal excitation', ...
          '(c) Stochastic excitation'};

for k = 1:3
    pname = panels{k};
    t     = ts.([pname '_t']);
    y_gt  = ts.([pname '_y_true_mm']);
    y_pr  = ts.([pname '_y_pred_mm']);

    % 截取前 2 秒
    t_max = min(2.0, t(end));
    idx = t <= t_max;

    subplot(3, 1, k);
    plot(t(idx), y_gt(idx), 'b-', 'LineWidth', 0.8); hold on;
    plot(t(idx), y_pr(idx), 'r--', 'LineWidth', 0.8);
    ylabel('h (mm)');
    title(titles{k}, 'FontSize', 9);
    set(gca, 'FontSize', 8, 'TickDir', 'in', 'Box', 'on');
    if k == 3
        xlabel('Time (s)');
    end
    if k == 1
        legend('Measured', 'Estimated', 'FontSize', 7, 'Location', 'northeast');
    end
    hold off;
end

exportgraphics(gcf, fullfile(result_dir, 'representative_case.pdf'), ...
    'ContentType', 'vector');
fprintf('已保存: representative_case.pdf\n');

%% ============================================================
%  图3: warm-up 图 (warmup_demo.pdf)
%  ============================================================

wu = load(fullfile(result_dir, 'warmup_data.mat'));

figure('Units', 'centimeters', 'Position', [2 2 12 7]);

% 截取前 1000 个样本
n_show = min(1000, length(wu.t));
idx = 1:n_show;

hold on;

% warm-up 阴影区域
warmup_t = wu.warmup_end_t;
y_lo = min(min(wu.gt_mm(idx)), min(wu.pred_mm(idx))) - 0.05;
y_hi = max(max(wu.gt_mm(idx)), max(wu.pred_mm(idx))) + 0.05;
fill([0 warmup_t warmup_t 0], [y_lo y_lo y_hi y_hi], ...
    [0.85 0.85 0.85], 'EdgeColor', 'none', 'FaceAlpha', 0.5);

% 数据曲线
plot(wu.t(idx), wu.gt_mm(idx), 'b-', 'LineWidth', 0.8);
plot(wu.t(idx), wu.pred_mm(idx), 'r--', 'LineWidth', 0.8);

% warm-up 结束标注
xline(warmup_t, 'k:', 'LineWidth', 0.8);
text(warmup_t + 0.0005, y_hi - 0.02, ...
    sprintf('n = 249 (%.2f ms)', warmup_t * 1000), ...
    'FontSize', 7, 'VerticalAlignment', 'top');

xlabel('Time (s)');
ylabel('h (mm)');
legend('Warm-up region', 'Measured', 'Estimated (fixed-point)', ...
    'FontSize', 7, 'Location', 'east');
set(gca, 'FontSize', 8, 'TickDir', 'in', 'Box', 'on');
ylim([y_lo y_hi]);
hold off;

exportgraphics(gcf, fullfile(result_dir, 'warmup_demo.pdf'), ...
    'ContentType', 'vector');
fprintf('已保存: warmup_demo.pdf\n');

fprintf('\n全部图表生成完成。\n');
