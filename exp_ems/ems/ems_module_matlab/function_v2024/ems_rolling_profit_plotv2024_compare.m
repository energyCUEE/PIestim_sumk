function [f_profit, f_Pnet, f_batt] = ems_rolling_profit_plotv2024_compare(all_sol, type)

    method_colors = lines(length(all_sol));
    method_names = {all_sol.methodname};

    f_profit = figure;
    t_profit = tiledlayout(4,4,'TileSpacing','loose','Padding','loose');
    ax = gobjects(1,16);

    % Loop over months
    for m = 3:12
        ax(m) = nexttile;
        hold on;
        for i = 1:length(all_sol)
            sol = all_sol(i).sol_rolling;
            PARAM = sol.PARAM;
            Pnet_ems = sol.Pnet;
            Pnet_actual = PARAM.NL_actual + sum(sol.Pchg,2) - sum(sol.Pdchg,2);
            res_hr = PARAM.Resolution / 60;
            profit_ems = res_hr * PARAM.normalize_factor * (max(0,-Pnet_ems).*PARAM.Sell_rate - max(0,Pnet_ems).*PARAM.Buy_rate);
            profit_actual = res_hr * PARAM.normalize_factor * (max(0,-Pnet_actual).*PARAM.Sell_rate - max(0,Pnet_actual).*PARAM.Buy_rate);

            % Change to negative profit
            profit_ems = -profit_ems;
            profit_actual = -profit_actual;

            cum_profit_ems = cumsum(profit_ems);
            cum_profit_actual = cumsum(profit_actual);

            idx = (month(sol.datetime) == m);
            switch type
                case 'ems'
                    month_profit = cumsum([0; diff(cum_profit_ems(idx))]);
                case 'actual'
                    month_profit = cumsum([0; diff(cum_profit_actual(idx))]);
                otherwise
                    warning('Select between actual and ems. Defaulting to actual.')
                    month_profit = cumsum([0; diff(cum_profit_actual(idx))]);
            end
            stairs(sol.datetime(idx), month_profit, 'Color', method_colors(i,:), 'LineWidth', 1.5);
        end
        title(['Month ', num2str(m)]);
        % ylabel('Cum. profit [USD]');
        ylabel('Cum. cost [USD]');

        grid on;
        hold off;
    end

    % Yearly overview
    ax(13) = nexttile([1 4]);
    hold on;
    for i = 1:length(all_sol)
        sol = all_sol(i).sol_rolling;
        PARAM = sol.PARAM;
        Pnet_ems = sol.Pnet;
        Pnet_actual = PARAM.NL_actual + sum(sol.Pchg,2) - sum(sol.Pdchg,2);

        res_hr = PARAM.Resolution / 60;
        profit_ems = res_hr * PARAM.normalize_factor * (max(0,-Pnet_ems).*PARAM.Sell_rate - max(0,Pnet_ems).*PARAM.Buy_rate);
        profit_actual = res_hr * PARAM.normalize_factor * (max(0,-Pnet_actual).*PARAM.Sell_rate - max(0,Pnet_actual).*PARAM.Buy_rate);

        profit_ems = -profit_ems;
        profit_actual = -profit_actual;

        cum_profit_ems = cumsum(profit_ems);
        cum_profit_actual = cumsum(profit_actual);

        cum_error_percent = ( cum_profit_ems(end) - cum_profit_actual(end) ) / cum_profit_actual(end) * 100;
        disp(['For ', all_sol(i).methodname,': Cumulative profit error = ', num2str(cum_error_percent), ' %'])

        switch type
            case 'ems'
                stairs(sol.datetime, cum_profit_ems, 'Color', method_colors(i,:), 'LineWidth', 1.5);
            case 'actual'
                stairs(sol.datetime, cum_profit_actual, 'Color', method_colors(i,:), 'LineWidth', 1.5);
            otherwise
                warning('Select between actual and ems. Defaulting to actual.')
                stairs(sol.datetime, cum_profit_actual, 'Color', method_colors(i,:), 'LineWidth', 1.5);
        end
        
    end
    title('Year 2024');
    xlabel('Time');
    % ylabel('Cum. profit [USD]');
    ylabel('Cum. cost [USD]');
    legend(method_names, 'Location', 'southoutside', 'Orientation', 'horizontal');
    grid on;
    hold off;

    sgtitle('Compare cumulative profit across all methods', 'FontWeight', 'bold');
    fontsize(0.45, 'centimeters');

    % Plot Pnet comparison
    f_Pnet = figure;
    % t_Pnet = tiledlayout(3,4,'TileSpacing','loose','Padding','loose');

    m = 3;
    idx = (month(sol.datetime) == m);
    hold on;
    for i = 1:length(all_sol)
        sol = all_sol(i).sol_rolling;
        PARAM = sol.PARAM;
        Pnet_ems = sol.Pnet;
        Pnet_actual = PARAM.NL_actual + sum(sol.Pchg,2) - sum(sol.Pdchg,2);

        switch type
            case 'ems'
                stairs(sol.datetime(idx), Pnet_ems(idx), 'Color', method_colors(i,:), 'LineWidth', 1.5);
            case 'actual'
                stairs(sol.datetime(idx), Pnet_actual(idx), 'Color', method_colors(i,:), 'LineWidth', 1.5);
            otherwise
                warning('Select between actual and ems. Defaulting to actual.')
                stairs(sol.datetime(idx), Pnet_actual(idx), 'Color', method_colors(i,:), 'LineWidth', 1.5);
        end  
    end
    hold off;
    xlabel('Time');
    ylabel('Power [kW]');
    legend(method_names, 'Location', 'southoutside', 'Orientation', 'horizontal');
    sgtitle(['Month ', num2str(m), ' - Compare P_{net} from all methods'], 'FontWeight', 'bold');
    fontsize(0.5,'centimeters');

    % for m = 3:12
    %     nexttile;
    %     hold on;
    %     for i = 1:length(all_sol)
    %         sol = all_sol(i).sol_rolling;
    %         PARAM = sol.PARAM;
    %         Pnet_ems = sol.Pnet;
    %         Pnet_actual = PARAM.NL_actual + sum(sol.Pchg,2) - sum(sol.Pdchg,2);
    %         idx = (month(sol.datetime) == m);
    %         switch type
    %             case 'ems'
    %                 stairs(sol.datetime(idx), Pnet_ems(idx), 'Color', method_colors(i,:), 'LineWidth', 1.5);
    %             case 'actual'
    %                 stairs(sol.datetime(idx), Pnet_actual(idx), 'Color', method_colors(i,:), 'LineWidth', 1.5);
    %             otherwise
    %                 warning('Select between actual and ems. Defaulting to actual.')
    %                 stairs(sol.datetime(idx), Pnet_actual(idx), 'Color', method_colors(i,:), 'LineWidth', 1.5);
    %         end  
    % 
    %     end
    %     title(['Month ', num2str(m)]);
    %     xlabel('Time');
    %     ylabel('Power [kW]');
    %     grid on;
    %     hold off;
    % end

    % legend(method_names, 'Location', 'southoutside', 'Orientation', 'horizontal');
    % sgtitle('Compare P_{net} from all methods', 'FontWeight', 'bold');
    % fontsize(0.5,'centimeters');


    % Plot Batt status comparison
    f_batt = figure;
    % t_Pnet = tiledlayout(3,4,'TileSpacing','loose','Padding','loose');

    m = 3;
    idx = (month(sol.datetime) == m);
    hold on;
    for i = 1:length(all_sol)
        sol = all_sol(i).sol_rolling;
        Pbatt = sum(sol.Pchg,2) - sum(sol.Pdchg,2);
        stairs(sol.datetime(idx), Pbatt(idx), 'Color', method_colors(i,:), 'LineWidth', 1.5);
    end
    hold off;
    xlabel('Time');
    ylabel('Batt Power [kW]');
    legend(method_names, 'Location', 'southoutside', 'Orientation', 'horizontal');
    sgtitle(['Month ', num2str(m), ' - Compare Battery from all methods'], 'FontWeight', 'bold');
    fontsize(0.5,'centimeters');

end
