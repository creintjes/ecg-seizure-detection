function [MultiLengthDiscordTable,BSF,BSF_loc] = MADRID_2_0(T,minL,maxL,stepSize,train_test_split,enable_output)
    %% Handle invalid inputs
    % Constant Regions
    if contains_constant_regions(T,minL)
        errorMessage = [...
            'BREAK: There is at least one region of length minL that is constant, or near constant.\n\n', ...
            'Whether such regions should be called "anomalies" depends on the context, but in any case they are trivial to discover and should not be reported as a success in algorithm comparison (see Wu & Keogh "Current Time Series Anomaly Detection Benchmarks are Flawed and are Creating the Illusion of Progress")\n\n', ...
            'To fix this issue:\n', ...
            '1) Choose a longer length for minL.\n', ...
            '2) Add a small amount of noise to the entire time series (this will probably result in the current constant sections becoming top discords)\n', ...
            '3) Add a small linear trend to the entire time series (this will probably result in the current constant sections becoming motifs, and not discords)\n', ...
            '4) Carefully edit the data to remove the constant sections'];
        
        error(errorMessage);
    end
    %%
    % User defined
    TS_length = length(T);

    % Model values
    factor = 1;
    if length(1:factor:(TS_length-train_test_split)) * length(ceil(minL/factor):stepSize:ceil(maxL/factor)) < 5000000
        % polynomial model (of order 6)
        p_1 = [-4.66922312132205e-45	1.54665628995475e-35	-1.29314859463985e-26 2.01592418847342e-18	-2.54253065977245e-11	9.77027495487874e-05	-1.27055582771851e-05];
        p_2 = [-3.79100071825804e-42	3.15547030055575e-33	-6.62877819318290e-25	2.59437174380763e-17	-8.10970871564789e-11	7.25344313152170e-05	4.68415490390476e-07];
    else
        % linear model
        p_1 = [3.90752957831437e-05	0];
        p_2 = [1.94005690535588e-05	0];
    end
    p_4 = [1.26834880558841e-05	0];
    p_8 = [1.42210521045333e-05	0];
    p_16 = [1.82290885539705e-05	0]; 
    % Prediction
    factor = 16;predicted_execution_time_16 = polyval(p_16,length(1:factor:(TS_length-train_test_split)) * length(ceil(minL/factor):stepSize:ceil(maxL/factor)));
    factor = 8;predicted_execution_time_8 = polyval(p_8, length(1:factor:(TS_length-train_test_split)) * length(ceil(minL/factor):stepSize:ceil(maxL/factor)));
    factor = 4;predicted_execution_time_4 = polyval(p_4, length(1:factor:(TS_length-train_test_split)) * length(ceil(minL/factor):stepSize:ceil(maxL/factor)));
    factor = 2;predicted_execution_time_2 = polyval(p_2, length(1:factor:(TS_length-train_test_split)) * length(ceil(minL/factor):stepSize:ceil(maxL/factor)));
    factor = 1;predicted_execution_time_1 = polyval(p_1, length(1:factor:(TS_length-train_test_split)) * length(ceil(minL/factor):stepSize:ceil(maxL/factor)));

    if predicted_execution_time_1 < 10
        % If MADRID thinks it can converge in under 10 seconds, just run the algorithm
        factor = 1;
        [MultiLengthDiscordTable,BSF,BSF_loc,~] = MADRID(T(1:factor:end),ceil(minL/factor),ceil(maxL/factor),stepSize,ceil(train_test_split/factor),enable_output);
    else
        % Otherwise, spend up to 5 seconds, to estimate the times needed
        % Scale predicted time by machine speed
        testData= Test_DataMADRID;
        factor = 16;
        [MultiLengthDiscordTable,BSF,BSF_loc,actual_measurement_16] = MADRID(testData(1:factor:end),ceil(64/factor),ceil(256/factor),1,ceil(50000/factor),0);
        scaling_factor = actual_measurement_16 / 0.654610000000000;
        predicted_execution_time_16_scaled = predicted_execution_time_16 * scaling_factor;
        predicted_execution_time_8_scaled = predicted_execution_time_8 * scaling_factor;
        predicted_execution_time_4_scaled = predicted_execution_time_4 * scaling_factor;
        predicted_execution_time_2_scaled = predicted_execution_time_2 * scaling_factor;
        predicted_execution_time_1_scaled = predicted_execution_time_1 * scaling_factor;
        

        disp(['1) Predicted execution time for MADRID 1 to 16: ', num2str(predicted_execution_time_16_scaled,'%.1f'), ' seconds'])
        disp(['2) Predicted execution time for MADRID 1 to 8: ', num2str(predicted_execution_time_8_scaled,'%.1f'), ' seconds'])
        disp(['3) Predicted execution time for MADRID 1 to 4: ', num2str(predicted_execution_time_4_scaled,'%.1f'), ' seconds'])
        disp(['4) Predicted execution time for MADRID 1 to 2: ', num2str(predicted_execution_time_2_scaled,'%.1f'), ' seconds'])
        disp(['5) Predicted execution time for MADRID 1 to 1: ', num2str(predicted_execution_time_1_scaled,'%.1f'), ' seconds'])
        disp('6) Quit');

        choice = input('Please pick an option: ');

        switch choice
            case 1
                disp('You chose MADRID 1 to 16.');
                factor = 16;
                if ceil(minL/factor) < 2
                    disp("MADRID cannot be executed properly because minL/16 < 2, please select an option again");
                    return;
                end
                if ceil(maxL/factor) < 2
                    disp("MADRID cannot be executed properly because maxL/16 < 2, please select an option again");
                    return;
                end
            case 2
                disp('You chose MADRID 1 to 8.');
                factor = 8;
                if ceil(minL/factor) < 2
                    disp("MADRID cannot be executed properly because minL/8 < 2, please select an option again");
                    return;
                end
                if ceil(maxL/factor) < 2
                    disp("MADRID cannot be executed properly because maxL/8 < 2, please select an option again");
                    return;
                end
            case 3
                disp('You chose MADRID 1 to 4.');
                factor = 4;
                if ceil(minL/factor) < 2
                    disp("MADRID cannot be executed properly because minL/4 < 2, please select an option again");
                    return;
                end
                if ceil(maxL/factor) < 2
                    disp("MADRID cannot be executed properly because maxL/84< 2, please select an option again");
                    return;
                end
            case 4
                disp('You chose MADRID 1 to 2.');
                factor = 2;
                if ceil(minL/factor) < 2
                    disp("MADRID cannot be executed properly because minL/2 < 2, please select an option again");
                    return;
                end
                if ceil(maxL/factor) < 2
                    disp("MADRID cannot be executed properly because maxL/2 < 2, please select an option again");
                    return;
                end
            case 5
                disp('You chose MADRID 1 to 1.');
                factor = 1;
                if ceil(minL/factor) < 2
                    disp("MADRID cannot be executed properly because minL/1 < 2, please select an option again");
                    return;
                end
                if ceil(maxL/factor) < 2
                    disp("MADRID cannot be executed properly because maxL/1 < 2, please select an option again");
                    return;
                end
            case 6
                disp('You chose to quit.');
                return;
            otherwise
                disp('Invalid choice. Please pick a valid option.');
                return;
        end

        [MultiLengthDiscordTable,BSF,BSF_loc,~] = MADRID(T(1:factor:end),ceil(minL/factor),ceil(maxL/factor),stepSize,ceil(train_test_split/factor),enable_output);
    end

end
function testData = Test_DataMADRID
    rng(123456789)
    Fs = 10000;
    t = 0:1/Fs:10;
    f_in_start = 50;
    f_in_end = 60;
    f_in = linspace(f_in_start, f_in_end, length(t));
    phase_in = cumsum(f_in/Fs);
    y = sin(2*pi*phase_in);
    y = y + randn(size(y))/12; % add a little noise
    EndOfTrain = floor(length(y)/2); % Here I assume the first half is train
    y(EndOfTrain + 1200:EndOfTrain + 1200+64)= y(EndOfTrain + 1200:EndOfTrain + 1200+64)+ randn(size(y(EndOfTrain + 1200:EndOfTrain + 1200+64)))/3;
    % add a medium anomly 
    y(EndOfTrain + 4180:EndOfTrain + 4180+160)= y(EndOfTrain + 4180:EndOfTrain + 4180+160)+ randn(size(y(EndOfTrain + 4180:EndOfTrain + 4180+160)))/4;
    % add a long anomly 
    y(EndOfTrain + 8200 : EndOfTrain + 8390 ) = y(EndOfTrain + 8200 : EndOfTrain + 8390 )*0.5; % add a long anomly 
    testData = y;
end
function [MultiLengthDiscordTable,BSF,BSF_loc,time_bf] = MADRID(T,minL,maxL,stepSize,train_test_split,enable_output)
    BSFseed = -inf; % This is for the first run if DAMP_topK 
    k=1;
    time_bf = 0;
    
    % Initialization
    MultiLengthDiscordTable = -inf(ceil((maxL+1-minL)/stepSize), length(T));
    BSF = zeros(ceil((maxL+1-minL)/stepSize), 1);
    BSF_loc = nan(ceil((maxL+1-minL)/stepSize), 1);
    
    
    % Data for creating convergence plots
    time_sum_bsf = [0,0];
    percent_sum_bsf = [0,0];
    
    tic;
    %%
    m_set = minL:stepSize:maxL;
    m_pointer = ceil(length(m_set)/2);
    m = m_set(m_pointer);
    [discord_score,position,Left_MP] = DAMP_2_0(T,m,train_test_split);
    MultiLengthDiscordTable(m_pointer,:) = Left_MP*(1/(2*sqrt(m)));
    BSF(m_pointer) = discord_score*(1/(2*sqrt(m)));
    BSF_loc(:) = position;

    
    m_pointer = 0;
    for m = m_set
        m_pointer = m_pointer+1;    
        if m_pointer == ceil(length(m_set)/2)
            continue;
        end
        i = position; 
        SubsequenceLength = m;
        if SubsequenceLength < 2
            break
        end
        if i+SubsequenceLength-1 > length(T)
            break
        end
        query = T(i:i+SubsequenceLength-1);
        % Use the brute force for the leftMP
        MultiLengthDiscordTable(m_pointer,i) = min( real(MASS_V2(T(1:i), query)))*(1/(2*sqrt(m))); 
        % Update the best so far discord score for current row
        BSF(m_pointer) = MultiLengthDiscordTable(m_pointer,i);
        BSF_loc(m_pointer) = i;
    end
    
   %%
   %%%%%%%%%%%%%%
    m_pointer = 1;
    m = m_set(m_pointer);
    [~,position_2,Left_MP] = DAMP_2_0(T,m,train_test_split);
    MultiLengthDiscordTable(m_pointer,:) = Left_MP*(1/(2*sqrt(m)));
    [BSF(m_pointer),BSF_loc(m_pointer)] = max(MultiLengthDiscordTable(m_pointer,:));

    if position_2 ~= position
        m_pointer = 0;
        for m = m_set
            m_pointer = m_pointer+1;    
            if m_pointer == ceil(length(m_set)/2) ||  m_pointer == 1
                continue;
            end
            i = position_2; 
            SubsequenceLength = m;
            if i+SubsequenceLength-1 > length(T)
                break
            end
            query = T(i:i+SubsequenceLength-1);
            % Use the brute force for the leftMP
            MultiLengthDiscordTable(m_pointer,i) = min( real(MASS_V2(T(1:i), query)))*(1/(2*sqrt(m))); 
            % Update the best so far discord score for current row
            [BSF(m_pointer),BSF_loc(m_pointer)] = max(MultiLengthDiscordTable(m_pointer,:));
        end
    end

    %%%%%%%%%%%%%%
    m_pointer = length(m_set);
    m = m_set(m_pointer);
    [~,position_3,Left_MP] = DAMP_2_0(T,m,train_test_split);
    MultiLengthDiscordTable(m_pointer,:) = Left_MP*(1/(2*sqrt(m)));
    [BSF(m_pointer),BSF_loc(m_pointer)] = max(MultiLengthDiscordTable(m_pointer,:));

    if position_3 ~= position_2 && position_3 ~= position
        m_pointer = 0;
        for m = m_set
            m_pointer = m_pointer+1;    
            if m_pointer == ceil(length(m_set)/2) ||  m_pointer == 1 || m_pointer == length(m_set)
                continue;
            end
            i = position_3; 
            SubsequenceLength = m;
            if i+SubsequenceLength-1 > length(T)
                break
            end
            query = T(i:i+SubsequenceLength-1);
            % Use the brute force for the leftMP
            MultiLengthDiscordTable(m_pointer,i) = min( real(MASS_V2(T(1:i), query)))*(1/(2*sqrt(m))); 
            % Update the best so far discord score for current row
            [BSF(m_pointer),BSF_loc(m_pointer)] = max(MultiLengthDiscordTable(m_pointer,:));
        end
    end
    %%
    initialization_time = toc;
    time_bf = time_bf+initialization_time;
    time_sum_bsf(end+1,:) = [initialization_time,sum(BSF)];
    percent_sum_bsf(end+1,:) = [progress_percentage(MultiLengthDiscordTable(:,train_test_split+1:end)),sum(BSF)];
    

    m_pointer = 0;
    for m = m_set
        m_pointer = m_pointer+1;    
        if m_pointer == ceil(length(m_set)/2) ||  m_pointer == 1 || m_pointer == length(m_set)
            continue;
        end
        tic;
        [Results BFS_for_i_plus_1 Left_MP]= DAMP_topK_new(T,train_test_split,m,k,0,max(BSFseed,BSF(m_pointer)));
        tmp = toc;
    
        time_bf = time_bf+tmp;
    
        BSF(m_pointer) = Results(1)*(1/(2*sqrt(m))); % only for k=1
        BSF_loc(m_pointer) = Results(2); % only for k=1
        MultiLengthDiscordTable(m_pointer,:) = Left_MP*(1/(2*sqrt(m)));
        time_sum_bsf(end+1,:) = [time_bf,sum(BSF)];
        percent_sum_bsf(end+1,:) = [progress_percentage(MultiLengthDiscordTable(:,train_test_split+1:end)),sum(BSF)];
        BSFseed = BFS_for_i_plus_1 - 0.000001;     
    end
    
    
    if enable_output
        disp(strcat("MADRID's elapsed time is ",num2str(time_bf)," seconds."))
        % create convergence plot
        figure;
        plot(time_sum_bsf(:,1),time_sum_bsf(:,2));        
        xlabel('Time');
        ylabel('Quality of solution');
        figure;
        plot(percent_sum_bsf(:,1),percent_sum_bsf(:,2));
        xlabel('Percentage of processed pixels');
        ylabel('Quality of solution');

        figure;
        MERLIN_STEM_PLOTindex = minL:stepSize:maxL;
        MERLIN_STEM_PLOTdiscordLoc = BSF_loc;
        MERLIN_STEM_PLOTdiscordVal = BSF;
        hold on;
        for i = 1 : k % loop over the top k anomalies
            stem3( MERLIN_STEM_PLOTindex(:), MERLIN_STEM_PLOTdiscordLoc(:,i) ,  MERLIN_STEM_PLOTdiscordVal(:,i)  )
        end
        set(gca,'Ylim',[1,length(T)])
        set(gca,'Xlim',[minL,maxL])
        set(gca,'xdir', 'reverse');
        T_normalized_for_plot = (T-min(T))/max(T);
        T_normalized_for_plot = T_normalized_for_plot * max(MERLIN_STEM_PLOTdiscordVal(:,1));
        plot3(  ones(size([1:length(T)]))* maxL , [1:length(T)],T_normalized_for_plot,'b' );
        plot3(  ones(size([1:train_test_split]))* maxL , [1:train_test_split],T_normalized_for_plot(1:train_test_split),'r' );
        xlabel('discord lengths');
        ylabel('positions in the input time series');
        zlabel('discord scores');
        view([59 68]);
    end

end
function [percentage] = progress_percentage(A)
    [rows, cols] = size(A);
    num_elements = rows * cols - sum(isinf(A(:)));
    percentage = num_elements / (rows * cols); 
end

function  [Results BFS_for_i_plus_1 Left_MP] = DAMP_topK_new(T,CurrentIndex,SubsequenceLength,discord_num,enable_output,BSFseed)
    
    % This is a special Matrix Profile, it only looks left (backwards in
    % time)
	Left_MP  = zeros(size(T));

    % Initialization 
    % The best discord score so far
    best_so_far = BSFseed;
    % A Boolean vector where 1 means execute the current iteration and 0
    % means skip the current iteration
    bool_vec = ones(1,length(T));
    % Lookahead indicates how long the algorithm has a delay
    lookahead = 2^nextpow2(16*SubsequenceLength);

    for i = CurrentIndex : (length(T)-SubsequenceLength+1)
        % Skip the current iteration if the corresponding boolean value is
        % 0, otherwise execute the current iteration
        if ~bool_vec(i)
            % We subtract a very small number here to avoid the pruned
            % subsequence having the same discord score as the real discord
            Left_MP(i) = Left_MP(i-1)-0.00001;
            continue
        end
        % Use the brute force for the left Matrix Profile value
        if i+SubsequenceLength-1 > length(T)
            break
        end
        
        % Initialization for classic DAMP
        % Approximate leftMP value for the current subsequence
        approximate_distance = inf;
        % X indicates how long a time series to look backwards
        X = 2^nextpow2(8*SubsequenceLength);
        % flag indicates if it is the first iteration of DAMP
        flag = 1;
        % expansion_num indicates how many times the search has been
        % expanded backward
        expansion_num = 0;
        query = T(i:i+SubsequenceLength-1);

        % Classic DAMP
        while approximate_distance >= best_so_far
            % Case 1: Execute the algorithm on the time series segment
            % farthest from the current subsequence 
            % Arrived at the beginning of the time series
            if i-X+1+(expansion_num * SubsequenceLength) < 1
                approximate_distance = min( real(MASS_V2(T(1:i), query)));
                Left_MP(i) = approximate_distance; 
                % Update the best discord so far
                if approximate_distance > best_so_far    
                    % The current subsequence is the best discord so far
                    best_so_far = approximate_distance;
                    Left_MP_copy = Left_MP;
                    for k= 1:discord_num
                        [best_so_far,idx_max]=max(Left_MP_copy);
                        discord_start = max(1,idx_max - floor(SubsequenceLength*0.5));
                        discord_end = max(1+floor(SubsequenceLength*0.5),idx_max + floor(SubsequenceLength*0.5));
                        Left_MP_copy(discord_start:discord_end)=-inf;
                    end
                end
                break
            else
                if flag == 1
                    % Case 2: Execute the algorithm on the time series
                    % segment closest to the current subsequence
                    flag = 0;
                    approximate_distance = min( real(MASS_V2(T(i-X+1:i), query))); 
                else 
                    % Case 3: All other cases
                    X_start = i-X+1+(expansion_num * SubsequenceLength);
                    X_end = i-(X/2)+(expansion_num * SubsequenceLength);
                    approximate_distance = min( real(MASS_V2(T(X_start:X_end), query))); 
                end

                if approximate_distance < best_so_far
                    % If a value less than the current best discord score
                    % exists on the distance profile, stop searching
                    Left_MP(i) = approximate_distance;
                    break
                else
                    % Otherwise expand the search
                    X = 2*X;
                    expansion_num = expansion_num+1;
                end
            end % end if
        end % end while
        
        % If lookahead is 0, then it is a pure online algorithm with no
        % pruning
        if lookahead ~= 0
            % Perform forward MASS for pruning 
            % The index at the beginning of the forward mass should be
            % avoided in the exclusion zone
            start_of_mass = i+SubsequenceLength;
            if start_of_mass > length(T) 
                start_of_mass = length(T);
            end
            end_of_mass = start_of_mass + lookahead - 1;
            if end_of_mass > length(T)
                end_of_mass = length(T);
            end
            % The length of lookahead should be longer than that of the
            % query
            if (end_of_mass - start_of_mass + 1) > SubsequenceLength
                distance_profile = real(MASS_V2(T(start_of_mass:end_of_mass), query));
                % Find the subsequence indices less than the best so far
                % discord score
                dp_index_less_than_BSF = find((distance_profile<best_so_far)==1);
                % Converting indexes on distance profile to indexes on time
                % series
                ts_index_less_than_BSF = dp_index_less_than_BSF + start_of_mass - 1;
                % update the Boolean vector
                bool_vec(ts_index_less_than_BSF) = 0;
            end    
        end
    end % end for
    

   Results = zeros(discord_num,2);
   BFS_for_i_plus_1 = [];
    % Print pruning rate
    PV = bool_vec(CurrentIndex:(length(T)-SubsequenceLength+1));
    PR = (length(PV)-sum(PV))/(length(PV));
    if enable_output
        disp(strcat("Pruning Rate: ",num2str(PR)));
    end

    % Print top K discords
    Left_MP_copy = Left_MP;
    for k= 1:discord_num
        [val,loc]=max(Left_MP_copy);
        if val == 0
            if enable_output
                disp(strcat("Only ",num2str(k-1)," discords are found."));
            end
            %%
            if k == 1
                BFS_for_i_plus_1(end+1) = -inf;
            end
            %%
            break
        end
        if enable_output
            disp(strcat("Predicted discord score/position","(top ",num2str(k),"): ",num2str(val),"/",num2str(loc)));
        end
        Results(k,:)=[val,loc];
        
        discord_start = max(1,loc);
        discord_end = max(1 + SubsequenceLength+1,loc + SubsequenceLength+1);
        BFS_for_i_plus_1(end+1) = min( real(MASS_V2(T(1:discord_start), T(discord_start:min(discord_start*2-1,discord_end))))); 
        Left_MP_copy(discord_start:discord_end)=-inf;
    end
    BFS_for_i_plus_1 = min(BFS_for_i_plus_1);

    if enable_output
        % Create the plot
	    figure; hold on; plot(Left_MP,'b'); plot(zscore(T)-2,'r'); 
   end
end
function [dist] = MASS_V2(x, y)
    %x is the data, y is the query
    m = length(y);
    n = length(x);

    %compute y stats -- O(n)
    meany = mean(y);
    sigmay = std(y,1);

    %compute x stats -- O(n)
    meanx = movmean(x,[m-1 0]);
    sigmax = movstd(x,[m-1 0],1);

    y = y(end:-1:1);%Reverse the query
    y(m+1:n) = 0; %aappend zeros

    %The main trick of getting dot products in O(n log n) time
    X = fft(x);
    Y = fft(y);
    Z = X.*Y;
    z = ifft(Z);

    dist = 2*(m-(z(m:n)-m*meanx(m:n)*meany)./(sigmax(m:n)*sigmay));
    dist = sqrt(dist);
end


function  [discord_score,position,Left_MP] = DAMP_2_0(T,SubsequenceLength,location_to_start_processing,varargin)
    % Set default values for parameters
    p = inputParser;            
    % Use our pre-defined default value if the user does not specify the
    % length of the lookahead
    addParameter(p,'lookahead',2^nextpow2(16*SubsequenceLength));  
    % Output is enabled by default
    addParameter(p,'enable_output',false);      
    parse(p,varargin{:});
    lookahead = p.Results.lookahead;
    enable_output = p.Results.enable_output;




    % Initialization 
    % This is a special Matrix Profile, it only looks left (backwards in
    % time)
	Left_MP  = -inf(size(T));
    Left_MP(1:location_to_start_processing) = nan;

    % The best discord score so far
    best_so_far = -inf;
    % A Boolean vector where 1 means execute the current iteration and 0
    % means skip the current iteration
    bool_vec = ones(1,length(T));


	% Handle the prefix to get a relatively high best so far discord score
    for i = location_to_start_processing : (location_to_start_processing+(16*SubsequenceLength))
        % Skip the current iteration if the corresponding boolean value is
        % 0, otherwise execute the current iteration
        if ~bool_vec(i)
            Left_MP(i) = Left_MP(i-1)-0.00001;
            continue
        end
        
        % Use the brute force for the left Matrix Profile value
        if i+SubsequenceLength-1 > length(T)
            break
        end
        query = T(i:i+SubsequenceLength-1);
        Left_MP(i) = min( real(MASS_V2(T(1:i), query))); 
        
        % Update the best so far discord score
        best_so_far = max(Left_MP);

        % If lookahead is 0, then it is a pure online algorithm with no
        % pruning
        if lookahead ~= 0
            % Perform forward MASS for pruning 
            % The index at the beginning of the forward mass should be
            % avoided in the exclusion zone
            start_of_mass = min(i+SubsequenceLength,length(T));
            end_of_mass = min((start_of_mass+lookahead-1),length(T));
            % The length of lookahead should be longer than that of the
            % query
            if (end_of_mass - start_of_mass + 1) > SubsequenceLength
                distance_profile = real(MASS_V2(T(start_of_mass:end_of_mass), query));
                % Find the subsequence indices less than the best so far
                % discord score
                dp_index_less_than_BSF = find((distance_profile<best_so_far)==1);
                % Converting indexes on distance profile to indexes on time
                % series
                ts_index_less_than_BSF = dp_index_less_than_BSF + start_of_mass - 1;
                % update the Boolean vector
                bool_vec(ts_index_less_than_BSF) = 0;
            end    
        end
    end

    % Remaining test data except for the prefix
    for i = (location_to_start_processing+(16*SubsequenceLength)+1) : (length(T)-SubsequenceLength+1)
        % Skip the current iteration if the corresponding boolean value is
        % 0, otherwise execute the current iteration
        if ~bool_vec(i)
            % We subtract a very small number here to avoid the pruned
            % subsequence having the same discord score as the real discord
            Left_MP(i) = Left_MP(i-1)-0.00001;
            continue
        end
        
        % Initialization for classic DAMP
        % Approximate leftMP value for the current subsequence
        approximate_distance = inf;
        % X indicates how long a time series to look backwards
        X = 2^nextpow2(8*SubsequenceLength);
        % flag indicates if it is the first iteration of DAMP
        flag = 1;
        % expansion_num indicates how many times the search has been
        % expanded backward
        expansion_num = 0;
        if i+SubsequenceLength-1 > length(T)
            break
        end
        query = T(i:i+SubsequenceLength-1);

        % Classic DAMP
        while approximate_distance >= best_so_far
            % Case 1: Execute the algorithm on the time series segment
            % farthest from the current subsequence 
            % Arrived at the beginning of the time series
            if i-X+1+(expansion_num * SubsequenceLength) < 1
                approximate_distance = min( real(MASS_V2(T(1:i), query)));
                Left_MP(i) = approximate_distance; 
                % Update the best discord so far
                if approximate_distance > best_so_far    
                    % The current subsequence is the best discord so far
                    best_so_far = approximate_distance;
                end
                break
            else
                if flag == 1
                    % Case 2: Execute the algorithm on the time series
                    % segment closest to the current subsequence
                    flag = 0;
                    approximate_distance = min( real(MASS_V2(T(i-X+1:i), query))); 
                else 
                    % Case 3: All other cases
                    X_start = i-X+1+(expansion_num * SubsequenceLength);
                    X_end = i-(X/2)+(expansion_num * SubsequenceLength);
                    approximate_distance = min( real(MASS_V2(T(X_start:X_end), query))); 
                end

                if approximate_distance < best_so_far
                    % If a value less than the current best discord score
                    % exists on the distance profile, stop searching
                    Left_MP(i) = approximate_distance;
                    break
                else
                    % Otherwise expand the search
                    X = 2*X;
                    expansion_num = expansion_num+1;
                end
            end % end if
        end % end while
        
        % If lookahead is 0, then it is a pure online algorithm with no
        % pruning
        if lookahead ~= 0
            % Perform forward MASS for pruning 
            % The index at the beginning of the forward mass should be
            % avoided in the exclusion zone
            start_of_mass = min(i+SubsequenceLength,length(T));
            end_of_mass = min((start_of_mass+lookahead-1),length(T));
            % The length of lookahead should be longer than that of the
            % query
            if (end_of_mass - start_of_mass + 1) > SubsequenceLength
                distance_profile = real(MASS_V2(T(start_of_mass:end_of_mass), query));
                % Find the subsequence indices less than the best so far
                % discord score
                dp_index_less_than_BSF = find((distance_profile<best_so_far)==1);
                % Converting indexes on distance profile to indexes on time
                % series
                ts_index_less_than_BSF = dp_index_less_than_BSF + start_of_mass - 1;
                % update the Boolean vector
                bool_vec(ts_index_less_than_BSF) = 0;
            end    
        end
    end % end for
   
    
    % Get pruning rate
    PV = bool_vec(location_to_start_processing:(length(T)-SubsequenceLength+1));
    PR = (length(PV)-sum(PV))/(length(PV));
    % Get top discord
    [discord_score,position] = max(Left_MP);

    % Outputs
    if enable_output
        disp("Results:")
        disp(strcat("Pruning Rate: ",num2str(PR)));
	    disp(strcat("Predicted discord score/position: ",num2str(discord_score),"/",num2str(position)));
        fprintf("\n* If you want to suppress the outputs, please call DAMP using the following format:\n" + ...
            ">> [discord_score,position] = DAMP_2_0(T,SubsequenceLength,location_to_start_processing,'enable_output',false)\n\n");
       
        % Create the plot
	    figure('name','UCR DAMP 2.0','NumberTitle','off'); 
        hold on;  
        set(gca,'ytick',[],'ycolor','w');
        plot(Left_MP/max(Left_MP),'b'); 
        plot((T-min(T))/(max(T)-min(T))+1.1,'r');
        hold off;
    end
end

function bool_vec = contains_constant_regions(T,SubsequenceLength)
    bool_vec = 0;
    constant_indices = find(vertcat(1,diff(T(:)),1));
    constant_length = max(diff(constant_indices));
    if constant_length >= SubsequenceLength || var(T) < 0.2
        bool_vec = 1;
    end
end