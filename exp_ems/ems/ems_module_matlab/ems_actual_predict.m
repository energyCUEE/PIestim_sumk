%% Load data and setup for rolling optimization (ROLLING)
clear all; close all; clc;

% ---- add the directory functions path ----
addpath('./function_v2024/');

use_actual = false;
modeforecastlist = {'upper', 'lower'}; % 'upper' or 'lower'
methodnamelist = {'sumk', 'cwc', 'qd', 'qr'};  % Use cell array for strings
% methodnamelist = {'sumk'}

% use_actual = true;
% modeforecastlist = {'upper'};
% methodnamelist = {'sumk'};

for modeforecast_index = 1:length(modeforecastlist)
    modeforecast = modeforecastlist{modeforecast_index};
    for method_index = 1:length(methodnamelist)
        methodname = methodnamelist{method_index};
        filename = sprintf('../input_data/netload_%s_forecast_%s.csv', modeforecast, methodname);
        
        % Battery parameters
        PARAM.battery.charge_effiency = [0.95]; %bes charge eff
        PARAM.battery.discharge_effiency = [0.95*0.93]; %  bes discharge eff note inverter eff 0.93-0.96
        % SMALL BATTERY
        % PARAM.battery.discharge_rate = [5]; % kW max discharge rate
        % PARAM.battery.charge_rate = [5]; % kW max charge rate
        % PARAM.battery.actual_capacity = [25]; % kWh soc_capacity 
        % MEDIUM BATTERY
        PARAM.battery.discharge_rate = [10]; % kW max discharge rate
        PARAM.battery.charge_rate = [10]; % kW max charge rate
        PARAM.battery.actual_capacity = [50]; % kWh soc_capacity 
        % LARGE BATTERY
        % PARAM.battery.discharge_rate = [20]; % kW max discharge rate
        % PARAM.battery.charge_rate = [20]; % kW max charge rate
        % PARAM.battery.actual_capacity = [100]; % kWh soc_capacity 
        PARAM.battery.initial = [50]; % userdefined int 0-100 %
        PARAM.battery.min = [20]; %min soc userdefined int 0-100 %
        PARAM.battery.max = [80]; %max soc userdefined int 0-100 %
        PARAM.battery.V_norminal = [(672+864)/2]; % V norminal voltage
        % PARAM.battery.soc_terminal = [40]; %min soc for the end of each day
        PARAM.battery.soc_terminal = [0]; %min soc for the end of each day
        %end of batt
        PARAM.battery.num_batt = length(PARAM.battery.actual_capacity);
        
        % time_horizon = 60;  % min
        time_horizon = 240;
        TOU_CHOICE = 'smart1';
        
        data = readtable(filename);
        data.Properties.VariableNames{'Datetime'} = 'datetime';
        data.datetime = datetime(data.datetime, 'InputFormat', 'dd-MMM-yyyy HH:mm:ss');
        
        % ---- save parameters ----
        PARAM.start_date = data.datetime(1);
        PARAM.Resolution = minutes(diff(data.datetime(1:2)));
        PARAM.Horizon = time_horizon; 
        PARAM.TOU_CHOICE = TOU_CHOICE;
        PARAM.use_actual = use_actual;
        PARAM.weight_energyfromgrid = 0;
        PARAM.weight_energycost = 0;
        PARAM.weight_profit = 1;
        PARAM.weight_multibatt = 0;  % recommend 1e-4
        PARAM.weight_chargebatt = 0; % recommend 4.6
        PARAM.weight_smoothcharge  = 0.5; % recommend 3
        PARAM.weight_Pnetref = 0;       % recommend 5 (zero)
        PARAM.weight_Pchgref = 0;       % recommend 5
        PARAM.weight_Pdchgref  = 0;     % recommend 5
        
        if PARAM.weight_energyfromgrid > 0
            mode_str = "energyfromgrid";    
        elseif PARAM.weight_energycost > 0
            mode_str = "energycost";
        elseif PARAM.weight_profit > 0
            mode_str = "profit";
        end
        
        % Initialize an empty struct for storing the solutions
        sol_rolling = struct('datetime', [], 'Pchg', [], 'Pdchg', [], 'Pnet', [], 'soc', [], 'PARAM', struct());
        sol_rolling.PARAM = PARAM;
        sol_rolling.PARAM.NL = [];
        sol_rolling.PARAM.NL_actual = [];
        sol_rolling.PARAM.NL_predict = [];
        sol_rolling.PARAM.Buy_rate = [];
        sol_rolling.PARAM.Sell_rate = [];
        
        % Rolling window loop
        i = 1;
        num_points_per_day = 24*60 / PARAM.Resolution; 
        num_day_in_data = floor(height(data)/num_points_per_day);
        
        % day_index = num_day_in_data; % Last date index
        day_index = 100; % Date with negative netload (with rolling day = 6)
        % day_index = 50; % Random date
        num_rolling_day = 3; % Manually run with specific rolling day
        
        % day_index = 1; % Run for the whole year
        % num_rolling_day = (num_day_in_data - day_index) + 1; % Run until last day
        % num_rolling_day = 1; % Manually run with specific rolling day
        % num_rolling_day = 1;
        
        current_start = data.datetime(num_points_per_day*(day_index-1)+1);
        
        while current_start <= data.datetime(num_points_per_day*(day_index + num_rolling_day - 1))
            fprintf('Running for window starting at %s\n', datestr(current_start));
            
            % Flags to check existence
            found_in_data = any(data.datetime == current_start);
        
            % Display check results
            if found_in_data
                disp('✅ current_start found in data.');
            else
                disp('❌ current_start NOT found in data.');
            end
                
            input_start_date = datestr(current_start, 'dd-mmm-yyyy HH:MM:SS');
        
            % Get data for current window
            [PARAM.start_date, resolution, PARAM.NL_actual, PARAM.NL_predict] = get_netload_HA_data(data, input_start_date, time_horizon);
            
            % --- Check if actual and predict lengths match ---
            if length(PARAM.NL_actual) ~= length(PARAM.NL_predict)
                fprintf('❌ Skipping %s due to length mismatch:\n', datestr(current_start));
                fprintf('   NL_actual: %d, NL_predict: %d\n', length(PARAM.NL_actual), length(PARAM.NL_predict));
                current_start = current_start + minutes(PARAM.Resolution);
                continue;
            end
        
            if use_actual
                PARAM.NL = PARAM.NL_actual;
            else
                PARAM.NL = PARAM.NL_predict;
            end
            [PARAM.Buy_rate, PARAM.Sell_rate] = getBuySellrate(PARAM.start_date, resolution, time_horizon, TOU_CHOICE);
            [buy_rate, sell_rate] = getBuySellrate(datetime(2023, 1, 1, 0, 0, 0), resolution, 60*24, TOU_CHOICE);
            % ---- Normalize Buy_rate and Sell_rate ----
            PARAM.normalize_factor = max([buy_rate; sell_rate]);
            PARAM.Buy_rate = PARAM.Buy_rate / PARAM.normalize_factor;
            PARAM.Sell_rate = PARAM.Sell_rate / PARAM.normalize_factor;
            
            if hour(PARAM.start_date) == 23 && minute(PARAM.start_date) == 0
                PARAM.battery.soc_terminal = [40]; %min soc for the end of each day
            else
                PARAM.battery.soc_terminal = [0]; %min soc for the end of each day
            end
            % Solve optimization
            sol = ems_econ_optv2024(PARAM);
            % Name fordername
            % save_folder = sprintf('./solution/sol_HA/sol_ems_three_days_3/%s/%s/sol_batches_%s/', methodname, mode_str, modeforecast);
            save_folder = sprintf('./solution/sol_HA/sol_ems/actual/%s/sol_batches/', mode_str);
            if ~exist(save_folder, 'dir')
                mkdir(save_folder);
            end
            save_filename_sol = sprintf('%ssol_%s.mat', save_folder, datestr(PARAM.start_date, 'yyyy-mm-dd_HHMMSS'));
            save(save_filename_sol, 'sol');
        
            % ---- set initial soc ----
            num_time_steps_five_minutes = 1;   % index of time 5min
            PARAM.battery.initial = sol.soc(2, :);
            
            % ---- Store the solution ----
            sol_rolling.datetime = [sol_rolling.datetime; PARAM.start_date];
            sol_rolling.Pchg = [sol_rolling.Pchg; sol.Pchg(1, :)];
            sol_rolling.Pdchg = [sol_rolling.Pdchg; sol.Pdchg(1, :)];
            sol_rolling.Pnet = [sol_rolling.Pnet; sol.Pnet(1, :)];
            sol_rolling.soc = [sol_rolling.soc; sol.soc(1, :)];
            sol_rolling.PARAM.NL = [sol_rolling.PARAM.NL; sol.PARAM.NL(1, :)];
            sol_rolling.PARAM.NL_actual = [sol_rolling.PARAM.NL_actual; sol.PARAM.NL_actual(1, :)];
            sol_rolling.PARAM.NL_predict = [sol_rolling.PARAM.NL_predict; sol.PARAM.NL_predict(1, :)];
            sol_rolling.PARAM.Buy_rate = [sol_rolling.PARAM.Buy_rate; sol.PARAM.Buy_rate(1, :)];
            sol_rolling.PARAM.Sell_rate = [sol_rolling.PARAM.Sell_rate; sol.PARAM.Sell_rate(1, :)];
            sol_rolling.PARAM.Resolution = PARAM.Resolution;
            sol_rolling.PARAM.normalize_factor = PARAM.normalize_factor;
            sol_rolling.PARAM.use_actual = PARAM.use_actual;
        
            % Move to next window
            current_start = current_start + minutes(resolution);
            i = i + 1;
        end
        
        % ---- Save results sol_rolling ----
        save_filename = sprintf('./solution/sol_HA/sol_ems_worachit_three_days_3/%s/%s/sol_rolling_%s_all.mat', methodname, mode_str, modeforecast);
        % save_filename = sprintf('./solution/sol_HA/sol_ems_worachit/actual/%s/sol_rolling_all.mat', mode_str);
        save(save_filename, "sol_rolling"); % Saved variable name: sol_rolling
        
        close all;
    end
end