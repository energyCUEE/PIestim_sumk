function sol = ems_econ_optv2024(PARAM)
    %%% This function is used to solve optimization problem, consisting 3 parts.
    %%% (I) Define optimization variables.
    %%% (II) Define constraints.
    %%% (III) Call the solver and save parameters.
    
    % Set optimization solving time.
    options = optimoptions('linprog','MaxTime',3600*60);
    
    if rem(PARAM.Horizon, PARAM.Resolution) % Check if the optimization horizon and resolution are compatible.
        error('horizon must be a multiple of resolution')            
    end
    if (PARAM.weight_energyfromgrid < 0 ) ||  (PARAM.weight_energycost < 0) || (PARAM.weight_profit < 0 ) || (PARAM.weight_multibatt  < 0) || (PARAM.weight_chargebatt < 0) || (PARAM.weight_smoothcharge < 0 )
        error('Weights must >= 0')
    end
    if  PARAM.weight_energyfromgrid*PARAM.weight_profit + PARAM.weight_energycost*PARAM.weight_energyfromgrid + PARAM.weight_profit*PARAM.weight_energycost > 0
        error('You can only choose one of the three energy objectives')   
    end
    if (PARAM.weight_multibatt > 0 ) && (PARAM.battery.num_batt == 1)
        error('The number of battery must >= 2 to use this objective')
    end
    length_optimvar = PARAM.Horizon/PARAM.Resolution; % Length of optimization variable.
    
    % Change the unit of Resolution from (minute => hour) to be used in Expense calculation.
    minutes_in_hour = 60;
    resolution_in_hour = PARAM.Resolution/minutes_in_hour;    
    Pnet =      optimvar('Pnet',length_optimvar,'LowerBound',-inf,'UpperBound',inf);
    % Pbatt =     optimvar('Pbatt',length_optimvar,PARAM.battery.num_batt,'LowerBound',-PARAM.battery.charge_rate,'UpperBound',PARAM.battery.discharge_rate);
    Pdchg =     optimvar('Pdchg',length_optimvar,PARAM.battery.num_batt,'LowerBound',0,'UpperBound',max(PARAM.battery.discharge_rate));
    Pchg =      optimvar('Pchg',length_optimvar,PARAM.battery.num_batt,'LowerBound',0,'UpperBound',max(PARAM.battery.charge_rate));
    soc =       optimvar('soc',length_optimvar+1,PARAM.battery.num_batt,'LowerBound',ones(length_optimvar+1,PARAM.battery.num_batt).*PARAM.battery.min,'UpperBound',ones(length_optimvar+1,PARAM.battery.num_batt).*PARAM.battery.max);
    
    % Define optimization variable corresponding to the main cost
    if PARAM.weight_energyfromgrid > 0
        % Define Pnet_pos is the upper bound of max(0, Pnet).
        Pnet_pos = optimvar('Pnet_pos', length_optimvar, 'LowerBound', 0, 'UpperBound', inf);

        total_obj = resolution_in_hour * sum(Pnet_pos, 1);       
    elseif PARAM.weight_energycost > 0
        % Define Pnet_pos is the upper bound of max(0, Pnet).
        Pnet_pos = optimvar('Pnet_pos', length_optimvar, 'LowerBound', 0, 'UpperBound', inf);

        total_obj = resolution_in_hour * sum(PARAM.Buy_rate.*Pnet_pos, 1);
    elseif PARAM.weight_profit > 0
        % Define Pnet_abs is the upper bound of abs(Pnet).
        Pnet_abs = optimvar('Pnet_abs', length_optimvar, 'LowerBound', -inf, 'UpperBound', inf);

        total_obj = resolution_in_hour * sum( (PARAM.Buy_rate - PARAM.Sell_rate)./2 .* Pnet_abs + (PARAM.Buy_rate + PARAM.Sell_rate)./2 .* Pnet, 1);
    end

    
    if PARAM.weight_multibatt > 0 % Add soc diff objective
        % Define optimvar for 'multibatt' objective
        % s = Upper bound of |SoC diff|/100
        % for 2 batt use batt difference for >= 3 batt use central soc
        if PARAM.battery.num_batt == 2
            s =         optimvar('s',length_optimvar,'LowerBound',0,'UpperBound',inf);
            total_obj = total_obj + PARAM.weight_multibatt*sum(s/100,'all'); % Add soc diff objective           
          
        elseif PARAM.battery.num_batt >= 3
            s =         optimvar('s',length_optimvar+1,PARAM.battery.num_batt,'LowerBound',0,'UpperBound',inf);
            central_soc = optimvar('central_soc',length_optimvar+1,'LowerBound',0,'UpperBound',inf);
            total_obj = total_obj + PARAM.weight_multibatt*sum(s/100,'all');           
        end
    end
    if PARAM.weight_chargebatt > 0  
        % Add term for 'chargebatt' objective
        total_obj = total_obj + PARAM.weight_chargebatt*( sum(sum(repmat(PARAM.battery.max,length_optimvar+1,1)-soc, 1)./(PARAM.battery.max - PARAM.battery.min)) );
        
    end
    if PARAM.weight_smoothcharge > 0 % Add non fluctuation charge and discharge objective
        % Define optimvars for 'smoothcharge' objective
        % upper_bound_Pchg is Upper bound of |Pchg(t)-Pchg(t-1)| / Pchg,max objective
        % upper_bound_Pdchg is Upper bound of |Pdchg(t)-Pdchg(t-1)| / Pdchg,max objective
        upper_bound_Pchg = optimvar('upper_bound_Pchg',length_optimvar-1,PARAM.battery.num_batt,'LowerBound',0,'UpperBound',inf);      
        upper_bound_Pdchg = optimvar('upper_bound_Pdchg',length_optimvar-1,PARAM.battery.num_batt,'LowerBound',0,'UpperBound',inf);        
        % Add non fluctuation charge and discharge objective.
        % Assume that the weight is equal for both Pchg and Pdchg.
        total_obj = total_obj + PARAM.weight_smoothcharge * ( sum(sum(upper_bound_Pchg, 1)./PARAM.battery.charge_rate) + sum(sum(upper_bound_Pdchg,1)./PARAM.battery.discharge_rate) );
       
    end

    if PARAM.weight_Pnetref > 0
        % Define upper_Pnet_HA is the upper bound of abs(Pnet,HA(t) − Pnet,ref(t)).
        upper_Pnet_HA = optimvar('upper_Pnet_HA', length_optimvar, 'LowerBound', -inf, 'UpperBound', inf);
        
        total_obj = total_obj + PARAM.weight_Pnetref * ( sum(upper_Pnet_HA) / max(PARAM.PV_capacity, PARAM.Load_capacity) );
    end
    if PARAM.weight_Pchgref > 0
        % Define upper_Pchg_HA is the upper bound of abs(Pchg,HA(t) − Pchg,ref(t)).
        upper_Pchg_HA = optimvar('upper_Pchg_HA', length_optimvar, PARAM.battery.num_batt, 'LowerBound', -inf, 'UpperBound', inf);
        
        total_obj = total_obj + PARAM.weight_Pchgref * ( sum(sum(upper_Pchg_HA, 1)./PARAM.battery.charge_rate) );
    end
    if PARAM.weight_Pdchgref > 0
        % Define upper_Pdchg_HA is the upper bound of abs(Pdchg,HA(t) − Pdchg,ref(t)).
        upper_Pdchg_HA = optimvar('upper_Pdchg_HA', length_optimvar, PARAM.battery.num_batt, 'LowerBound', -inf, 'UpperBound', inf);
        
        total_obj = total_obj + PARAM.weight_Pdchgref * ( sum(sum(upper_Pdchg_HA, 1)./PARAM.battery.discharge_rate) );
    end

    %-------------------------- assign cost function -----------------------
    prob =      optimproblem('Objective', total_obj, 'ObjectiveSense', 'minimize');
    
    % Define optimization variable corresponding to the main cost
    if PARAM.weight_energyfromgrid > 0
        %--constraint for epigraph form of Pnet_pos >= Pnet 
        prob.Constraints.epicons1 = Pnet_pos >= Pnet;    
    elseif PARAM.weight_energycost > 0
        %--constraint for epigraph form of Pnet_pos >= Pnet 
        prob.Constraints.epicons1 = Pnet_pos >= Pnet;
    elseif PARAM.weight_profit > 0
        %--constraint for epigraph form of Pnet_abs >= Pnet and Pnet_abs >= -Pnet
        prob.Constraints.epicons1 = Pnet_abs >= Pnet;
        prob.Constraints.epicons2 = Pnet_abs >= -Pnet;
    end

    if PARAM.weight_multibatt > 0 % Add soc diff objective
        % Define optimvar for 'multibatt' objective
        % s = Upper bound of |SoC diff|
        % for 2 batt use batt difference for >= 3 batt use central soc
        if PARAM.battery.num_batt == 2                     
            prob.Constraints.battdeviate1 = soc(2:length_optimvar+1,1) - soc(2:length_optimvar+1,2) <= s;
            prob.Constraints.battdeviate2 = -s <= soc(2:length_optimvar+1,1) - soc(2:length_optimvar+1,2);
        elseif PARAM.battery.num_batt >= 3     
            prob.Constraints.battdeviate1 = soc - central_soc*ones(1,PARAM.battery.num_batt) <= s ;
            prob.Constraints.battdeviate2 = -s <= soc - central_soc*ones(1,PARAM.battery.num_batt);
        end
    end  
    if PARAM.weight_smoothcharge > 0 % Add non fluctuation charge and discharge objective           
        % %-- Constraint non fluctuating charge and discharge
        % abs(Pchg(t)-Pchg(t-1)) <= upper_bound_Pchg
        prob.Constraints.non_fluct_Pchg_con1 = Pchg(1:end-1,:)-Pchg(2:end,:) <= upper_bound_Pchg;
        prob.Constraints.non_fluct_Pchg_con2 = -upper_bound_Pchg <= Pchg(1:end-1,:)-Pchg(2:end,:);
        % abs(Pdchg(t)-Pdchg(t-1)) <= upper_bound_Pdchg
        prob.Constraints.non_fluct_Pdchg_con1 = Pdchg(1:end-1,:)-Pdchg(2:end,:) <= upper_bound_Pdchg;
        prob.Constraints.non_fluct_Pdchg_con2 = -upper_bound_Pdchg <= Pdchg(1:end-1,:)-Pdchg(2:end,:);
    end
    
    if PARAM.weight_Pnetref > 0
        % %-- Constraint
        % abs(Pnet,HA(t) − Pnet,ref(t)) <= upper_Pnet_HA
        prob.Constraints.epicon_Pnetref1 = Pnet - PARAM.Pnetref <= upper_Pnet_HA;
        prob.Constraints.epicon_Pnetref2 = -( Pnet - PARAM.Pnetref ) <= upper_Pnet_HA;
    end
    if PARAM.weight_Pchgref > 0
        % %-- Constraint
        % abs(Pchg,HA(t) − Pchg,ref(t)) <= upper_Pchg_HA
        prob.Constraints.epicon_Pchgref1 = Pchg - PARAM.Pchgref <= upper_Pchg_HA;
        prob.Constraints.epicon_Pchgref2 = -( Pchg - PARAM.Pchgref ) <= upper_Pchg_HA;
    end
    if PARAM.weight_Pdchgref > 0
        % %-- Constraint
        % abs(Pchg,HA(t) − Pchg,ref(t)) <= upper_Pchg_HA
        prob.Constraints.epicon_Pdchgref1 = Pdchg - PARAM.Pdchgref <= upper_Pdchg_HA;
        prob.Constraints.epicon_Pdchgref2 = -( Pdchg - PARAM.Pdchgref ) <= upper_Pdchg_HA;
    end

    % Constraint part
    %--Pbatt constraint
    % prob.Constraints.Pdchg_cons1 = 0 <= Pdchg;  
    % prob.Constraints.Pdchg_cons2 = Pbatt <= Pdchg;
    % prob.Constraints.Pchg_cons1 = 0 <= Pchg;
    % prob.Constraints.Pchg_cons2 = -Pbatt <= Pchg;
    % prob.Constraints.Pbatt_cons = Pbatt == Pdchg - Pchg;

    %--Pnet constraint
    prob.Constraints.powercons = Pnet == PARAM.NL + sum(Pchg,2) - sum(Pdchg,2);
    
    %end of static constraint part
    
    %--soc dynamic constraint
    soccons = optimconstr(length_optimvar+1,PARAM.battery.num_batt);
    
    soccons(1,1:PARAM.battery.num_batt) = soc(1,1:PARAM.battery.num_batt)  == PARAM.battery.initial ;
    for j = 1:PARAM.battery.num_batt
        soccons(2:length_optimvar+1,j) = soc(2:length_optimvar+1,j)  == soc(1:length_optimvar,j) + ...
                                 (PARAM.battery.charge_effiency(:,j)*100*resolution_in_hour/PARAM.battery.actual_capacity(:,j))*Pchg(1:length_optimvar,j) ...
                                    - (resolution_in_hour*100/(PARAM.battery.discharge_effiency(:,j)*PARAM.battery.actual_capacity(:,j)))*Pdchg(1:length_optimvar,j);
        
    end
    prob.Constraints.soccons = soccons;
    
     %--min soc for the end of each day constraint
    if all(PARAM.battery.soc_terminal ~= 0)
        time_steps_per_day = 60/PARAM.Resolution * 24;
        num_days = PARAM.Horizon/(24*60);
        if num_days >= 1
            end_of_day_indices = linspace(time_steps_per_day, num_days*time_steps_per_day, num_days);
            for i = 1:length(end_of_day_indices)
                prob.Constraints.(['EndOfDayConstraint' num2str(i)]) = soc(end_of_day_indices(i), :) >= PARAM.battery.soc_terminal;
            end
        elseif hour(PARAM.start_date) == 23 && minute(PARAM.start_date) == 0
            prob.Constraints.('EndOfDayConstraint') = soc(end-1, :) >= PARAM.battery.soc_terminal;
        end
    end

    %---solve for optimal sol
    [sol, fval, exitflag] = solve(prob,'Options',options);
    % [sol, fval, exitflag] = solve(prob);
    sol.fval = fval;
    sol.exitflag = exitflag;
    sol.PARAM = PARAM;
end