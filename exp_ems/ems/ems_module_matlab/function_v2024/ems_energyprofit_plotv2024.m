function [f,t] = ems_energyprofit_plotv2024(sol)
    PARAM = sol.PARAM;
    %----------------prepare solution for plotting
    net_load = PARAM.PL - PARAM.PV;
    % end of prepare for solution for plotting
    resolution_HR = PARAM.Resolution/60; % (min) Resolution in minutes
    profit = resolution_HR * PARAM.normalize_factor * ( max(0,-sol.Pnet).*PARAM.Sell_rate - max(0,sol.Pnet).*PARAM.Buy_rate );
    start_date = datetime(PARAM.start_date);
    end_date = (datetime(PARAM.start_date)+minutes(PARAM.Horizon));
    vect = start_date:minutes(PARAM.Resolution):end_date;
    vect = vect(1:end-1);
    k = PARAM.Horizon/PARAM.Resolution; % length of variable
    f = figure;
    t = tiledlayout(2,2,'TileSpacing','loose','Padding','loose');
    
    nexttile
    stairs(vect,PARAM.PV,'LineWidth',1.2) 
    grid on
    hold on
    stairs(vect,PARAM.PL,'LineWidth',1.2)
    stairs(vect, net_load, '-k', 'LineWidth', 1.2);
    ylabel('Power (kW)')
    legend('Solar','Load', 'Net Load','Location','northeastoutside')
    title('Solar generation and load consumption power')
    xlabel('Time');
    if PARAM.Horizon <= 60
        xticks(start_date:minutes(10):end_date);
    elseif PARAM.Horizon <= 360  % 6 hour
        xticks(start_date:minutes(30):end_date);
    elseif PARAM.Horizon <= 1440  % 24 hour
        xticks(start_date:hours(1):end_date);
    else
        xticks(start_date:hours(24):end_date);
    end
    datetick('x', 'mmm dd, HH:MM', 'keepticks');
    hold off
    
    nexttile
    stairs(vect,sol.soc(1:k,1),'-k','LineWidth',1.5)
    ylabel('SoC (%)')
    ylim([0 100])
    grid on
    hold on
    stairs(vect,[PARAM.battery.min(1)*ones(k,1),PARAM.battery.max(1)*ones(k,1)],'--m','HandleVisibility','off','LineWidth',1.2)
    hold on
    yyaxis right
    stairs(vect,sol.Pchg(:,1),'-b','LineWidth',1.2)
    hold on 
    stairs(vect,sol.Pdchg(:,1),'-r','LineWidth',1.2)    
    legend('Soc','P_{chg}','P_{dchg}','Location','northeastoutside')
    ylabel('Power (kW)')
    title('State of charge (SoC)')
    xlabel('Time');
    if PARAM.Horizon <= 60
        xticks(start_date:minutes(10):end_date);
    elseif PARAM.Horizon <= 360  % 6 hour
        xticks(start_date:minutes(30):end_date);
    elseif PARAM.Horizon <= 1440  % 24 hour
        xticks(start_date:hours(1):end_date);
    else
        xticks(start_date:hours(24):end_date);
    end
    datetick('x', 'mmm dd, HH:MM', 'keepticks');
    
    nexttile;
    grid on;
    hold on;
    stairs(vect, sol.Pnet(:, 1), '-k', 'LineWidth', 1.2);
    stairs(vect, sol.Pchg(:, 1), '-b', 'LineWidth', 1.2);
    stairs(vect, sol.Pdchg(:, 1), '-r', 'LineWidth', 1.2);
    ylabel('Power (kW)');
    yyaxis right;
    stairs(vect, PARAM.normalize_factor * PARAM.Buy_rate, '-m', 'LineWidth', 1.2);
    stairs(vect, PARAM.normalize_factor * PARAM.Sell_rate, '-c', 'LineWidth', 1.2);
    ylabel('TOU (THB)');
    ylim([1 8])
    title('Power consumption and TOU rates over time');
    xlabel('Time');
    if PARAM.Horizon <= 60
        xticks(start_date:minutes(10):end_date);
    elseif PARAM.Horizon <= 360  % 6 hour
        xticks(start_date:minutes(30):end_date);
    elseif PARAM.Horizon <= 1440  % 24 hour
        xticks(start_date:hours(1):end_date);
    else
        xticks(start_date:hours(24):end_date);
    end
    datetick('x', 'mmm dd, HH:MM', 'keepticks');
    legend('P_{net}', 'P_{chg}', 'P_{dchg}', 'Buy rate', 'Sell rate','Location', 'northeastoutside');
    
    nexttile
    stairs(vect,profit,'-b','LineWidth',1)
    ylim([0 50])
    ylabel('Profit (THB)')
    hold on
    yyaxis right
    stairs(vect,cumsum(profit),'-r','LineWidth',1.5)
    ylabel('Cum. profit (THB)')
    title('With EMS 1') 
    legend('Profit','Cum. profit','Location','northeastoutside') 
    grid on
    % ylim([0 3500])
    xlabel('Time');
    if PARAM.Horizon <= 60
        xticks(start_date:minutes(10):end_date);
    elseif PARAM.Horizon <= 360  % 6 hour
        xticks(start_date:minutes(30):end_date);
    elseif PARAM.Horizon <= 1440  % 24 hour
        xticks(start_date:hours(1):end_date);
    else
        xticks(start_date:hours(24):end_date);
    end
    datetick('x', 'mmm dd, HH:MM', 'keepticks');
    hold off

    fontsize(0.35,'centimeters')
end