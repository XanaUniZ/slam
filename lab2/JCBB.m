%-------------------------------------------------------
function H = JCBB (prediction, observations, compatibility)
% 
%-------------------------------------------------------
global Best;
global configuration;

Best.H = zeros(1, observations.m);

JCBB_R (prediction, observations, compatibility, [], 1);

H = Best.H;
configuration.name = 'JCBB';

%-------------------------------------------------------
function JCBB_R (prediction, observations, compatibility, H, i)
% 
%-------------------------------------------------------
global Best;
global configuration;

if i > observations.m % leaf node?
    if pairings(H) > pairings(Best.H) % did better?
        Best.H = H;
    end
else
    % complete JCBB here
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
            JCBB_R(prediction, observations, compatibility, [H j], i+1);

        end
    end

    if pairings(H) + observations.m - i > pairings(Best.H)
        JCBB_R(prediction, observations, compatibility, [H 0], i+1)
    end
end

%-------------------------------------------------------
% 
%-------------------------------------------------------
function p = pairings(H)

p = length(find(H));

%-------------------------------------------------------
% Joint Compatibility function
%-------------------------------------------------------
% function jc = joint_compatibility(H, i, j, prediction, observations, compatibility)
%     H = [0, 2, 4, 1, 0, 5];
%     P_H = H*prediction.P*H.T + observations.R(H);
%     z_H = observations.z(H);
%     h_H = prediction.h(H);
%     % D_H = 
% 
%     if D_H <= chi2(2)
%         jc = 1;
%     else
%         jc = 0;
% end