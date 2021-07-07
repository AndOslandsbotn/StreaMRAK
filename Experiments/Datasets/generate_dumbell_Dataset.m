clear all; close all; clc;

%% Creating Simple Data, can be ignored for general algorithm
% data is n x d matrix X

%high dim spheres with 2D plane connecting 
d=5;
%N1 = 40000; m1=d; %change to make spheres at ends higher dimensional
%N2 = 40000; m2=d; %change to make spheres at ends higher dimensional
N1 = 1000000; m1=d; %change to make spheres at ends higher dimensional
N2 = 1000000; m2=d; %change to make spheres at ends higher dimensional
distBetween = 2;
%Nmiddle = 40000; mMiddle = 2;
Nmiddle = 1000000; mMiddle = 2;
thickMiddle = 1;

x1 = randn(N1,m1);
x1 = bsxfun(@times,x1,1./sqrt(sum(x1.^2,2))); % normalise to ball of radius 1
[~,ind] = sort(x1(:,1));
x1 = x1(ind,:);

x2 = randn(N2,m2);
x2 = bsxfun(@times,x2,1./sqrt(sum(x2.^2,2)));
x2 = bsxfun(@plus,x2,[distBetween+2,zeros(1,m2-1)]); % We move the second sphere a distance of 2
%along first axis, from the first sphere
x2 = [x2 zeros(N2,m1-m2)];
[~,ind] = sort(x2(:,1));
x2 = x2(ind,:);

% Generate a plane
x3 = rand(Nmiddle,mMiddle);
[~,ind] = sort(x3(:,1));
x3 = x3(ind,:);
x3 = bsxfun(@times,x3,[2+distBetween,thickMiddle*ones(1,mMiddle-1)]);
x3 = bsxfun(@minus,x3,[0,thickMiddle/2*ones(1,mMiddle-1)]);
index1 = sqrt(sum(x3.^2,2))<=1;
index2 = sqrt(sum( bsxfun(@minus,x3,[2+distBetween,zeros(1,mMiddle-1)]).^2,2))<=1;
x3(index1 | index2,:) = [];
x3 = [x3 zeros(size(x3,1),m1 - mMiddle)];
[~,ind] = sort(x3(:,1));
x3 = x3(ind,:);

%testing = x3;
%scatter3(testing(:,1),testing(:,2),ones(size(testing(:,2))))

X = [x1;x2;x3];
%scatter(x3(:,1),x3(:,2));
%scatter3(X(:,1),X(:,2),X(:,3),20);
%axis image

% we split in test and train datasets
% Cross varidation (train: 75%, test: 25%)
randperm_idx = randperm(length(X));
X(:,:) = X(randperm_idx, :);

n = 1000;
figure()
scatter3(X(1:n, 1), X(1:n, 2), X(1:n, 3))

rng('default');
cv = cvpartition(size(X,1),'HoldOut',0.25);
idx = cv.test;

% Separate to training and test data
Xtr = X(~idx,:);
Xts  = X(idx,:);


filenampath = strcat(pwd, '\Dumbell');
mkdir Dumbell

filename = 'twoblobTr.csv'
csvwrite(fullfile(filenampath,filename), Xtr)

filename = 'twoblobTs.csv'
csvwrite(fullfile(filenampath,filename), Xts)

