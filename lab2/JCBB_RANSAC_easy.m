%-------------------------------------------------------
function H = JCBB_RANSAC (prediction, observations, compatibility)
% 
%-------------------------------------------------------
global Best;
global configuration;

Best.H = zeros(1, observations.m);

JCBB_RANSAC_R (prediction, observations, compatibility, [], 1);

H = Best.H;
configuration.name = 'JCBB_RANSAC';

%-------------------------------------------------------
function JCBB_RANSAC_R (prediction, observations, compatibility, H, i)
% 
%-------------------------------------------------------
global Best;
global configuration;

if i > observations.m % leaf node?
    if pairings(H) > pairings(Best.H) % did better?
        Best.H = H;
    end
else
    % Define the number of attempts
    z = 0.01;
    w = 0.8;
    n = observations.m;
    fprintf('Number of attempts = %d\n', ceil(log(z)/log(1-w^n)));
    n_attempts = min(prediction.n, ceil(log(z)/log(1-w^n)));
    fprintf('Number of attempts = %d\n', n_attempts);
    fprintf('Number features    = %d\n', prediction.n);
    fprintf('\n');
    % Select t random positions without replacement
    % ind_comp = find(compatibility.ic(i,:));
    % ind_comp=ind_comp(randperm(size(ind_comp,2), n_attempts));
    % random_positions = ind_comp;
    random_positions = randperm(prediction.n, n_attempts);
    
    % complete JCBB here
    for j = random_positions
        ind_comp = compatibility.ic(i,j); 
        if ~isempty(H)
            %joint_comp = joint_compatibility(H,i,j,prediction, observations, compatibility);
            H_temp = [H j];
            [joint_comp, d2] = jointly_compatible(prediction, observations, H_temp);
        else
            joint_comp = 1;
        end
        % joint_comp = smth(H,i,j); 
        if ind_comp && joint_comp
            JCBB_RANSAC_R(prediction, observations, compatibility, [H j], i+1);

        end
    end

    if pairings(H) + observations.m - i > pairings(Best.H)
        JCBB_RANSAC_R(prediction, observations, compatibility, [H 0], i+1)
    end
end

%-------------------------------------------------------
% 
%-------------------------------------------------------
function p = pairings(H)

p = length(find(H));
