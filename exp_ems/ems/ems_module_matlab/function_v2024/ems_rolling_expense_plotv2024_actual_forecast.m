function [f_expense, f_Pnet_actual, f_Pnet_forecast, f_Pnet_mix] = ems_rolling_expense_plotv2024_actual_forecast(sol_rolling_actual,sol_rolling_forecast)
    
    PARAM_actual = sol_rolling_actual.PARAM;
    PARAM_forecast = sol_rolling_forecast.PARAM;
    
    Pnet_ideal = sol_rolling_actual.Pnet;
    Pnet_ems = sol_rolling_forecast.Pnet;
    Pnet_actual = PARAM_forecast.PL_actual - PARAM_forecast.PV_actual + sum(sol_rolling_forecast.Pchg,2) - sum(sol_rolling_forecast.Pdchg,2);
    net_load_actual = PARAM_actual.PL_actual - PARAM_actual.PV_actual;

    resolution_HR = PARAM_actual.Resolution/60; % (min) Resolution in minutes
    expense_ideal = resolution_HR * PARAM_actual.normalize_factor * ( max(0,Pnet_ideal).*PARAM_actual.Buy_rate );
    expense_ems = resolution_HR * PARAM_forecast.normalize_factor * ( max(0,Pnet_ems).*PARAM_forecast.Buy_rate );
    expense_actual = resolution_HR * PARAM_forecast.normalize_factor * ( max(0,Pnet_actual).*PARAM_forecast.Buy_rate );
    expense_noems = resolution_HR * PARAM_actual.normalize_factor * ( max(0,net_load_actual).*PARAM_actual.Buy_rate );
    cum_expense_ideal = cumsum(expense_ideal);
    cum_expense_ems = cumsum(expense_ems);
    cum_expense_actual = cumsum(expense_actual);
    cum_expense_noems = cumsum(expense_noems);
    
    cum_error_percent = ( cum_expense_actual(end) - cum_expense_ideal(end) ) / cum_expense_ideal(end) * 100;

    disp(['Cumulative expense error = ', num2str(cum_error_percent), ' %']);
    
    
    % -----------------------------
    % Plot graph cumulative expense
    % -----------------------------
    f_expense = figure;
    t_expense = tiledlayout(4,4,'TileSpacing','loose','Padding','loose');
    
    % Create an array to hold axes handles
    ax = gobjects(1,16);
    
    for m = 1:12
        % Extract data for each month
        idx = (month(sol_rolling_actual.datetime) == m);
        
        % Reset cumsum at the start of each month
        month_cum_expense_ideal = cumsum([0; diff(cum_expense_ideal(idx))]); 
        month_cum_expense_ems = cumsum([0; diff(cum_expense_ems(idx))]); 
        month_cum_expense_actual = cumsum([0; diff(cum_expense_actual(idx))]); 
        month_cum_expense_noems = cumsum([0; diff(cum_expense_noems(idx))]); 
    
        % Subplot for Expenditure Comparison
        ax(m) = nexttile;
        stairs(sol_rolling_actual.datetime(idx), month_cum_expense_ideal, 'b', 'LineWidth', 1.5);
        hold on;
        stairs(sol_rolling_forecast.datetime(idx), month_cum_expense_ems, 'r', 'LineWidth', 1.5);
        stairs(sol_rolling_forecast.datetime(idx), month_cum_expense_actual, 'g', 'LineWidth', 1.5);
        stairs(sol_rolling_actual.datetime(idx), month_cum_expense_noems, 'k', 'LineWidth', 1.5);
        title(['Month ', int2str(m)]);
        ylabel('Cum. expense [THB]');
        grid on;
        hold off;
    end
    
    % % Set the same y-axis limits for the first 3 rows (tiles 1 to 12)
    % idx_1to12 = 1:12;
    % y_mins = arrayfun(@(a) min(ax(a).YLim), idx_1to12);
    % y_maxs = arrayfun(@(a) max(ax(a).YLim), idx_1to12);
    % y_lim = [min(y_mins), max(y_maxs)];
    % set(ax(idx_1to12), 'YLim', y_lim);
    
    
    % Merge the last row (tile 13) to plot a full-year cumulative expense
    ax(13) = nexttile([1 4]);
    stairs(sol_rolling_actual.datetime, cum_expense_ideal, 'b', 'LineWidth', 1.5);
    hold on;
    stairs(sol_rolling_forecast.datetime, cum_expense_ems, 'r', 'LineWidth', 1.5);
    stairs(sol_rolling_forecast.datetime, cum_expense_actual, 'g', 'LineWidth', 1.5);
    stairs(sol_rolling_actual.datetime, cum_expense_noems, 'k', 'LineWidth', 1.5);
    title('Year 2024');
    xlabel('Time');
    ylabel('Cum. expense [THB]');
    grid on;
    hold off;
    
    % Create a global legend outside the subplots
    lgd = legend({'Ideal', 'EMS', 'Actual', 'No EMS'}, 'Location', 'southoutside', 'Orientation', 'horizontal');
    lgd.Layout.Tile = 'south';
    
    sgtitle('Compare cumulative expense with actual and forecast data', 'FontWeight', 'bold');
    
    fontsize(0.45,'centimeters');

    % ----------------------
    % Plot graph Pnet_actual
    % ----------------------
    f_Pnet_actual = figure;
    t_Pnet_actual = tiledlayout(3,4,'TileSpacing','loose','Padding','loose');

    for m = 1:12
        idx = (month(sol_rolling_actual.datetime) == m);
        
        nexttile;
        stairs(sol_rolling_actual.datetime(idx), Pnet_ideal(idx), 'b', 'LineWidth', 1.5);
        title(['Month ', int2str(m)]);
        xlabel('Time');
        ylabel('Power [kW]');
        grid on;
    end
    sgtitle('P_{net} with objective energycost with actual data and optimization solution', 'FontWeight', 'bold');
    
    fontsize(0.5,'centimeters')

    % ------------------------
    % Plot graph Pnet_forecast
    % ------------------------
    f_Pnet_forecast = figure;
    t_Pnet_forecast = tiledlayout(3,4,'TileSpacing','loose','Padding','loose');

    for m = 1:12
        idx = (month(sol_rolling_forecast.datetime) == m);
        
        nexttile;
        stairs(sol_rolling_forecast.datetime(idx), Pnet_ems(idx), 'r', 'LineWidth', 1.5);
        title(['Month ', int2str(m)]);
        xlabel('Time');
        ylabel('Power [kW]');
        grid on;
    end
    sgtitle('P_{net} with objective energycost with forecast data and optimization solution', 'FontWeight', 'bold');
    
    fontsize(0.5,'centimeters')

    % -------------------
    % Plot graph Pnet_mix
    % -------------------
    f_Pnet_mix = figure;
    t_Pnet_mix = tiledlayout(3,4,'TileSpacing','loose','Padding','loose');

    for m = 1:12
        idx = (month(sol_rolling_forecast.datetime) == m);
        
        nexttile;
        stairs(sol_rolling_forecast.datetime(idx), Pnet_actual(idx), 'g', 'LineWidth', 1.5);
        title(['Month ', int2str(m)]);
        xlabel('Time');
        ylabel('Power [kW]');
        grid on;
    end
    sgtitle('P_{net} with objective energycost with actual data and forecast optimization solution', 'FontWeight', 'bold');

    fontsize(0.5,'centimeters')
end
