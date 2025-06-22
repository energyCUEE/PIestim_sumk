%% Read the data
clear all; close all; clc;

% ---- add the directory functions path ----
addpath('./function_v2024/');
% methodnamelist = {'sumk', 'cwc', 'qd', 'qr'};  % Use cell array for strings
methodnamelist = {'qr', 'qd', 'cwc', 'sumk'};  % Use cell array for strings
type = 'ems';
mode_str = 'profit';
ideal_filename = sprintf('./solution/sol_HA/sol_ems/actual/%s/sol_rolling_all.mat', mode_str);
data_ideal = load(ideal_filename);
ideal_sol = data_ideal.sol_rolling;

PARAM = ideal_sol.PARAM;
Pnet_ideal = ideal_sol.Pnet;
res_hr = PARAM.Resolution / 60;

negprofit_ideal = - res_hr * PARAM.normalize_factor * (max(0,-Pnet_ideal).*PARAM.Sell_rate - max(0,Pnet_ideal).*PARAM.Buy_rate);
cum_negprofit_ideal = cumsum(negprofit_ideal);

cum_negprofit_upper_all = [];
cum_negprofit_lower_all = [];
for i = 1:length(methodnamelist)
    methodname = methodnamelist{i};
    filename = sprintf('./solution/sol_HA/sol_ems/%s/%s/sol_rolling_upper_all.mat', methodname, mode_str);
    data = load(filename);
    % all_sol_upper(i).sol_rolling = data.sol_rolling;
    % all_sol_upper(i).methodname = methodname;
    
    PARAM = data.sol_rolling.PARAM;
    res_hr = PARAM.Resolution/60;
    switch type
        case 'ems'
            Pnet_upper = data.sol_rolling.Pnet;
        case 'actual'
            Pnet_upper = PARAM.NL_actual + data.sol_rolling.Pchg - data.sol_rolling.Pdchg;
        otherwise
            warning('Select between actual and ems. Defaulting to actual.')
    end
   
    negprofit_upper = - res_hr * PARAM.normalize_factor * (max(0,-Pnet_upper).*PARAM.Sell_rate - max(0,Pnet_upper).*PARAM.Buy_rate);
    cum_negprofit_upper = cumsum(negprofit_upper);
    cum_negprofit_upper_all = [cum_negprofit_upper_all cum_negprofit_upper];

    filename = sprintf('./solution/sol_HA/sol_ems/%s/%s/sol_rolling_lower_all.mat', methodname, mode_str);
    data = load(filename);
    % all_sol_lower(i).sol_rolling = data.sol_rolling;
    % all_sol_lower(i).methodname = methodname;

    PARAM = data.sol_rolling.PARAM;
    res_hr = PARAM.Resolution/60;
    
    switch type
        case 'ems'
            Pnet_lower = data.sol_rolling.Pnet;
        case 'actual'
            Pnet_lower = PARAM.NL_actual + data.sol_rolling.Pchg - data.sol_rolling.Pdchg;
        otherwise
            warning('Select between actual and ems. Defaulting to actual.')
    end

    negprofit_lower = - res_hr * PARAM.normalize_factor * (max(0,-Pnet_lower).*PARAM.Sell_rate - max(0,Pnet_lower).*PARAM.Buy_rate);
    cum_negprofit_lower = cumsum(negprofit_lower);
    cum_negprofit_lower_all = [cum_negprofit_lower_all cum_negprofit_lower];
end
% 
% [f_profit, f_Pnet, f_batt] = ems_rolling_profit_plotv2024_compare_withactual(all_sol_upper, all_sol_lower, actual_sol, type);

result_array = [cum_negprofit_upper_all(end,:) ; cum_negprofit_lower_all(end,:)];
result_array = (result_array - cum_negprofit_ideal(end))*100/cum_negprofit_ideal(end)

fprintf('\\begin{table}[ht]\n');
fprintf('\\centering\n');
fprintf('\\renewcommand{\\arraystretch}{1.2}\n');
fprintf('\\begin{tabular}{lcccc}\n');
fprintf('\\hline\n');
fprintf('\\textbf{Deviation from ideal} & \\text{QR} & \\text{QD} & $\\text{CWC}_{\\text{Shri}}$ & Sum-$k$ \\\\\n');
fprintf('\\hline\n');
fprintf('Pessimistic (\\%%) & %.1f & %.1f & %.1f & %.1f \\\\\n', result_array(1, :));
fprintf('Optimistic (\\%%) & %.1f & %.1f & %.1f & %.1f \\\\\n', result_array(2, :));
fprintf('\\hline\n');
fprintf('\\end{tabular}\n');
fprintf('\\caption{Deviation from ideal for each method.}\n');
fprintf('\\label{tab:deviation_from_ideal}\n');
fprintf('\\end{table}\n');

%%
h = figure(1);
methodnamelist = {'qr', 'qd', 'cwc', 'sumk'};
methodnamelist_disp = {'QR', 'QD', 'CWC$_{\mathrm{Shri}}$', 'Sum-$k$'};
datetimeplotindex = datetime(data.sol_rolling.datetime);
% colorlist = {"#EDB120", "#A2142F", "#77AC30", "#0072BD"};
colorlist = {"#F1A400", "#D2042D", "#18B400", "#0047AB"};
hold on

currency_conversion = 0.0298;
% Plot the ideal cumulative negative profit with dashed black line
plot(datetimeplotindex, cum_negprofit_ideal/(currency_conversion*1e4), 'k', 'DisplayName', 'Ideal', 'LineWidth',2);

% Plot upper and lower cumulative negative profits for each method
for i = 1:length(methodnamelist)
    plot(datetimeplotindex, cum_negprofit_upper_all(:, i)/(currency_conversion * 1e4), 'Color', colorlist{i}, 'LineWidth', 1.5, ...
        'DisplayName', [methodnamelist_disp{i} ' Upper']);
    plot(datetimeplotindex, cum_negprofit_lower_all(:, i)/(currency_conversion*1e4), 'Color', colorlist{i}, 'LineStyle', '--', 'LineWidth', 1.5, ...
        'DisplayName', [methodnamelist_disp{i} ' Lower']);
end

datetick('x', 'mmmm', 'keepticks');

% Capture current tick labels
xticks_now = get(gca, 'XTick');
xticklabels_now = get(gca, 'XTickLabel');

% % Remove the last label (e.g., "January")
xticklabels_now = [
    '         ';
    xticklabels_now
];


% Apply the updated labels
set(gca, 'XTickLabel', xticklabels_now);

legend('show', 'Interpreter', 'latex', 'Location', 'northwest', 'FontSize', 14);
xlabel('Time', 'FontSize', 14)
ylabel('Cumulative Net Electricity Cost (x 10^{4} THB)', 'FontSize', 14)
ylim([0, 14])
title('Cumulative Net Electricity Cost: Ideal vs Methods', 'FontSize', 16)
grid on
hold off

%% 
upper_last_cost = cum_negprofit_upper_all(end, :)/(currency_conversion * 1e4);
lower_last_cost = cum_negprofit_lower_all(end, :)/(currency_conversion * 1e4);
interval_cost = upper_last_cost - lower_last_cost;

%% Plot graph rolling for three days
clear all; close all; clc;

% ---- add the directory functions path ----
addpath('./function_v2024/');

methodname = 'sumk';
modeforecastlist = {'upper', 'lower'};
savenamelist = {'pessimistic', 'optimistic'};
% colorlist = {'b', 'r'};
colorlist = {"#0072BD", "#D95319"};
mode_str = "profit";
type = 'ems';
casenamelsit = {'Pessimistic', 'Optimistic'};

h = figure;
set(h, 'Units', 'Inches', 'Position', [1, 1, 8, 8]);
% t = tiledlayout(5,2,'TileSpacing','loose','Padding','loose');

for i = 1:length(modeforecastlist)
    modeforecast = modeforecastlist{i};
    filename = sprintf('./solution/sol_HA/sol_ems_three_days/%s/%s/sol_rolling_%s_all.mat',methodname, mode_str, modeforecast);
    data = load(filename);
   
    Pnet = data.sol_rolling.Pnet;
    Pnet_actual = data.sol_rolling.PARAM.NL_actual + data.sol_rolling.Pchg - data.sol_rolling.Pdchg;
    resolution_HR = data.sol_rolling.PARAM.Resolution/60; % (min) Resolution in minutes
    datetimeplotindex =  datetime(data.sol_rolling.datetime);

    % Displaye unique date
    dt_dates = dateshift(datetimeplotindex, 'start', 'day');
    unique_dates = unique(dt_dates);
    disp(unique_dates);
            
    % % Subplot 1: Netload forecast
    % nexttile;
    subplot(4, 1, 1)
    hold on;
    stairs(datetimeplotindex, data.sol_rolling.PARAM.NL_predict, 'Color', colorlist{i}, 'LineWidth', 1.5, 'DisplayName', casenamelsit{i});
    % stairs(datetime, data_ideal.sol_rolling.PARAM.NL_predict, 'b', 'LineWidth', 1.5);
    legend('show');
    for j = 1:length(unique_dates)-2
        xline(unique_dates(j+1), '--k', 'LineWidth', 0.5, 'HandleVisibility','off');
    end
    % yline(0, '-', 'Color', [0.5 0.5 0.5], 'LineWidth', 0.5, 'HandleVisibility','off');
    ylim([-5, 15]);
    ylabel('Power [kW]');
    % Set XTick at 00:00 and 12:00
    dt_day_start = dateshift(datetimeplotindex(1), 'start', 'day');
    dt_day_end = dateshift(datetimeplotindex(end), 'start', 'day');
    
    xticks_full = dt_day_start : hours(6) : dt_day_end; % Every 12 hours (00:00, 12:00, 00:00, etc.)
    xticks(xticks_full);
    
    % Set XTick labels
    datetick('x', 'HH:MM', 'keepticks');
    xlabel('Time');
    title('Net load forecast', 'fontsize', 15);
    grid on;
   
   
    % Subplot 2: Pnet
    % nexttile;
    subplot(4, 1, 2)
    hold on;
    stairs(datetimeplotindex, Pnet, 'Color', colorlist{i}, 'LineWidth', 1.5, 'DisplayName', casenamelsit{i});
    % legend('show');
    for j = 1:length(unique_dates)-2
        xline(unique_dates(j+1), '--k', 'LineWidth', 0.5, 'HandleVisibility','off');
    end
    % stairs(datetime, Pnet_actual, 'b', 'LineWidth', 1.5);
    % legend('Pnet', 'Pnet_{actual}')
    ylim([-5, 25]);
    ylabel('Power [kW]');
    % Set XTick at 00:00 and 12:00
    dt_day_start = dateshift(datetimeplotindex(1), 'start', 'day');
    dt_day_end = dateshift(datetimeplotindex(end), 'start', 'day');
    
    xticks_full = dt_day_start : hours(6) : dt_day_end; % Every 12 hours (00:00, 12:00, 00:00, etc.)
    xticks(xticks_full);
    
    % Set XTick labels
    datetick('x', 'HH:MM', 'keepticks');
    xlabel('Time');
    title('P_{net}', 'fontsize', 15);
    grid on;
    
    % Subplot 3: Pchg
    % nexttile;
    subplot(4, 1, 3)
    % figure;
    hold on;
    stairs(datetimeplotindex, sum(data.sol_rolling.Pchg - data.sol_rolling.Pdchg, 2), 'Color', colorlist{i}, 'LineWidth', 1.5, 'DisplayName', casenamelsit{i});
    % stairs(datetime, sum(ideal_sol.Pchg, 2), 'b', 'LineWidth', 1.5);
    % legend('show');
    for j = 1:length(unique_dates)-2
        xline(unique_dates(j+1), '--k', 'LineWidth', 0.5, 'HandleVisibility','off');
    end
    ylim([-10, 12]);
    ylabel('Power [kW]');
    % Set XTick at 00:00 and 12:00
    dt_day_start = dateshift(datetimeplotindex(1), 'start', 'day');
    dt_day_end = dateshift(datetimeplotindex(end), 'start', 'day');
    
    xticks_full = dt_day_start : hours(6) : dt_day_end; % Every 12 hours (00:00, 12:00, 00:00, etc.)
    xticks(xticks_full);
    
    % Set XTick labels
    datetick('x', 'HH:MM', 'keepticks');
    xlabel('Time');
    % title('P_{dchg} - P_{chg}');
    title('Battery power (Charge > 0, Discharge < 0)', 'fontsize', 15);
    grid on;
    
    % % Subplot 5: SoC
    % nexttile;
    subplot(4, 1, 4)
    hold on;
    stairs(datetimeplotindex, data.sol_rolling.soc, 'Color', colorlist{i}, 'LineWidth', 1.5, 'DisplayName', casenamelsit{i});
    % stairs(datetime, data_ideal.sol_rolling.soc, 'b', 'LineWidth', 1.5);
    % legend('show');
    for j = 1:length(unique_dates)-2
        xline(unique_dates(j+1), '--k', 'LineWidth', 0.5, 'HandleVisibility','off');
    end
    ylim([10, 90]);
    ylabel('%');
    % Set XTick at 00:00 and 12:00
    dt_day_start = dateshift(datetimeplotindex(1), 'start', 'day');
    dt_day_end = dateshift(datetimeplotindex(end), 'start', 'day');
    
    xticks_full = dt_day_start : hours(6) : dt_day_end; % Every 12 hours (00:00, 12:00, 00:00, etc.)
    xticks(xticks_full);
    
    % Set XTick labels
    datetick('x', 'HH:MM', 'keepticks');
    xlabel('Time');
    title('Battery SOC', 'fontsize', 15);
    grid on;
end

set(h,'Units','Inches');
pos = get(h,'Position');
set(h,'PaperPositionMode','auto','PaperUnits','Inches','PaperSize',[pos(3), pos(4)])
savename = savenamelist{i};
savefilename = sprintf('emsrolling_sumk_compared_nattanon', savename);
% print(h,savefilename,'-dpdf','-r0')
print(h,savefilename,'-depsc')



