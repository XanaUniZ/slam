%-------------------------------------------------------
function H = JCBB_RANSAC_NN (prediction, observations, compatibility)
% 
%-------------------------------------------------------
global Best;
global configuration;

Best.H = zeros(1, observations.m);

z = 0.01;
w = 0.8;
b = 4;

n_attempts = ceil(log(z)/log(1-w^b));

for i=0:n_attempts
    % We should somehow shuffle the observations every attempt
    % or find some way to pick a subset
    JCBB_RANSAC_R_NN (prediction, observations, compatibility, [], 1, b);
end

H = Best.H;
configuration.name = 'JCBB_RANSAC_NN';

%-------------------------------------------------------
function JCBB_RANSAC_R_NN (prediction, observations, compatibility, H, i, b)
% 
%-------------------------------------------------------
global Best;
global configuration;

if pairings(H) == b 
    H = NN_for_JCCB([prediction, observations, compatibility, H, i+1]);
    if pairings(H) > pairings(Best.H) % did better?
        Best.H = H;
    end
else
    for j = 1:prediction.n
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
            JCBB_RANSAC_R_NN(prediction, observations, compatibility, [H j], i+1);
        end
    end
end

function NN_for_JCCB()
% This should use the simple nearest neighbor rule given hypothesis H, that is,
% finding pairings that are compatible with the first b features using NN

%-------------------------------------------------------
% 
%-------------------------------------------------------
function p = pairings(H)

p = length(find(H));
