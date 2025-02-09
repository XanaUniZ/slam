function H = SINGLES (prediction, observations, compatibility)
%-------------------------------------------------------
% University of Zaragoza
% Authors:  J. Neira, J. Tardos
%-------------------------------------------------------
%-------------------------------------------------------
global chi2;
global configuration;

H = zeros(1, observations.m);

% You have observations.m observations, and prediction.n
% predicted features.
%
% For every observation i, check whether it has only one neighbour,
% say feature j, and whether that feature j  has only that one neighbour
% observation i.  If so, H(i) = j.
%
% You will need to check the compatibility.ic matrix
% for this:
%
% compatibility.ic(i,j) = 1 if observation i is a neighbour of
% feature j.

for i = 1:observations.m
    matched = false;
    idx_neighbours_j = find(compatibility.ic (i,:));
    if length(idx_neighbours_j) == 1
        j = idx_neighbours_j(1);
        idx_neighbours_i = find(compatibility.ic (:,j));
        if length(idx_neighbours_j) == 1
            H(i)=j;
            matched = 1;
        end 
    end

    if ~matched
        H(i) = 0;
    end
end
            
configuration.name = 'SINGLES';
