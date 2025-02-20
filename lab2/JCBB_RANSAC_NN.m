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
    perm_idx = randperm(observations.m);
    JCBB_RANSAC_R_NN (prediction, observations, compatibility, [], 1, b, perm_idx);
end

H = Best.H;
configuration.name = 'JCBB_RANSAC_NN';

%-------------------------------------------------------
function JCBB_RANSAC_R_NN (prediction, observations, compatibility, H, i, b, perm_idx)
% 
%-------------------------------------------------------
global Best;
global configuration;

if pairings(H) == b 
    H = NN_for_JCCB(prediction, observations, compatibility, H, i, perm_idx);
    if pairings(H) > pairings(Best.H) % did better?
        inv_perm(perm_idx) = 1:length(perm_idx);
        Best.H = H(inv_perm);
    end
else
    for j = 1:prediction.n
        ind_comp = compatibility.ic(perm_idx(i),j); 
        if ~isempty(H)
            H_temp = [H j];
            [joint_comp, d2] = jointly_compatible_idx(prediction, observations, H_temp, perm_idx);
        else
            joint_comp = 1;
        end
        if ind_comp && joint_comp
            JCBB_RANSAC_R_NN(prediction, observations, compatibility, [H j], i+1, b, perm_idx);
        end
    end
end

%-------------------------------------------------------
function H = NN_for_JCCB(prediction, observations, compatibility, H, ind, perm_idx)
% 
%-------------------------------------------------------
global chi2;

H_nn = [];

for i = ind:observations.m
    D2 = compatibility.d2 (perm_idx(i), :);
    [D2_sorted, idx_near] = sort(D2, 'descend');

    found = false;
    for nearest = 1:length(idx_near)
        if D2_sorted(nearest) <= chi2(2)
            H_temp = [H idx_near(nearest)];
            if jointly_compatible_idx(prediction, observations, H_temp, perm_idx)
                H_nn = H_temp;
                found = true;
                break;
            end
        else
            H_nn = [H_nn 0];
            found = true;
            break;
        end
    end

    if ~found
        H_nn = [H_nn 0];
    end
end

H = [H H_nn];


function p = pairings(H)

p = length(find(H));

function [answer, d2] = jointly_compatible_idx (prediction, observations, H, idxs)
%-------------------------------------------------------
% University of Zaragoza
% Authors:  J. Neira, J. Tardos
%-------------------------------------------------------
%-------------------------------------------------------
global chi2;

d2 = joint_mahalanobis2_idx (prediction, observations, H, idxs);
dof = 2*length(find(H));
%dof = 2*observations.m;

answer = d2 < chi2(dof);

function [d2k, Hk, Ck, hk, zk, Rk] = joint_mahalanobis2_idx (prediction, observations, H, idxs)
%-------------------------------------------------------
%-------------------------------------------------------

% Compute joint distance for a hypothesis
[kk,i, j] = find(H);

[ix, iy, indi] = obs_rows(idxs(i));
[jx, jy, indj] = obs_rows(j);

zk = observations.z(indi);
hk = prediction.h(indj);
Rk = observations.R(indi,indi);
Ck = prediction.HPH(indj,indj) + Rk;
Hk = prediction.H(indj,:);
d2k = mahalanobis (zk - hk, Ck);
