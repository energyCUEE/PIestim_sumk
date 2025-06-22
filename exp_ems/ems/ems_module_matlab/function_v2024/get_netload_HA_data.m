function [start_date, resolution, NL_actual, NL_predict] = get_netload_HA_data(data, input_start_date, time_horizon)
    
    % Convert to datetime, extract start date, compute sampling time (resolution), and calculate total time horizon.
    datetime_vals = datetime(data.datetime, 'InputFormat', 'dd-mmm-yyyy HH:MM:SS');
    resolution = minutes(diff(datetime_vals(1:2)));
    start_date = datetime(input_start_date) + minutes(resolution);
    start_idx = find(datetime_vals == start_date, 1);
    if isempty(start_idx)
        error('Specified start_date not found in the data.');
    end
    end_date = start_date + minutes(time_horizon);

    % Select actual data in range [start_date, start_date + time_horizon)
    selected_actual_idx = (datetime_vals >= start_date) & (datetime_vals < end_date);

    % This solar profile is emulated from EE building which has the installation capacity
    % of 8 kWp, the PV generation power is scaled up to the desired capacity
    % source_capacity = 8; % kW PV installation capacity of source
    % desired_PVcapacity = 8; % Add by Worachit (???)XXXXXXXXX
    % PV_scale_factor = desired_PVcapacity / source_capacity; % Scale up from source to desired capacity (kW)

    % Compute both actual and predicted values

    NL_actual = data.netload_kW(selected_actual_idx);

    % PV_predict = PV_scale_factor * data{start_idx, startsWith(data.Properties.VariableNames, 'netload_kW_ahead')}';
    % PL_predict = data{start_idx, startsWith(data.Properties.VariableNames, 'Ltotnext')}';
    % NL_predict = data(selected_actual_idx, 3:end) % Check again
   
    start_idx_predict = find(datetime_vals == datetime(input_start_date), 1);
    NL_predict = data{start_idx_predict, 3:end}';
end
