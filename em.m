clear all; close all;

% X is the dataset size 100x2
%   Column 1: sepal length
%   Column 2: sepal width
X = dlmread('simple_iris_dataset.dat');  % Size=100x2
N = length(X);  % N=100

% Initialization - take 2 random samples from data set
u1 = X(randi([1,N]),:);
u2 = X(randi([1,N]),:);

cov1 = cov(X);
cov2 = cov(X);

prior1 = 0.5;
prior2 = 0.5;

% Misc. initialization
idx_c1 = zeros(50,1);
idx_c2 = zeros(50,1);

W1 = zeros(100,1);
W2 = zeros(100,1);

M1 = repmat(u1, N, 1);
M2 = repmat(u2, N, 1);

XM1 = X-M1;
XM2 = X-M2;

% W1 and W2 are vectors that eventually should contain each data point's
% membership grade relative to Gaussian 1 and Gaussian 2
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% You can define some error measure smaller than some epsilon to stop
% the iteration.  But for now just run for 250 iterations
for itr = 1:250
  % E-STEP
  likelihood1 = sum(-0.5 * XM1 * inv(cov1) .* XM1, 2);
  likelihood2 = sum(-0.5 * XM2 * inv(cov2) .* XM2, 2);

  posterior1 = likelihood1 * prior1;
  posterior2 = likelihood2 * prior2;

  normalization = posterior1 + posterior2;

  W1 = posterior1 ./ normalization;
  W2 = posterior2 ./ normalization;

  % M-STEP
  u1 = sum([W1 W1] .* X) / sum(W1);
  u2 = sum([W2 W2] .* X) / sum(W2);

  M1 = repmat(u1, N, 1);
  M2 = repmat(u2, N, 1);

  XM1 = X-M1;
  XM2 = X-M2;

  cov1 = (([W1 W1] .* XM1).' * XM1) / sum(W1);
  cov2 = (([W2 W2] .* XM2).' * XM2) / sum(W2);
end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
figure; hold on;

title('Clustering with EM algorithm');
xlabel('Sepal Length');
ylabel('Sepal Width');

% Hard clustering assignment – W1, W2 (100x1)

idx_c1 = find(W1 > W2);
idx_c2 = find(W1 <= W2);

ctr1 = mean(X(W1>W2,:));
ctr2 = mean(X(W1<=W2,:));

% idx_c1 is a vector containing the indices of the points in X
% that belong to cluster 1 (Mx1)
% idx_c2 is a vector containing the indices of the points in X
% that belong to cluster 2 (N-M x 1)
% Plot clustered data with two different colors
plot(X(idx_c1,1),X(idx_c1,2),'r.','MarkerSize',12)
plot(X(idx_c2,1),X(idx_c2,2),'b.','MarkerSize',10)

% Plot centroid of each cluster – ctr1, ctr2  (1x2)
plot(ctr1(:,1),ctr1(:,2), 'kx', 'MarkerSize',12,'LineWidth',2);
plot(ctr2(:,1),ctr2(:,2), 'ko', 'MarkerSize',12,'LineWidth',2);
