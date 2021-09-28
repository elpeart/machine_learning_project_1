clear
y = -10.5:.1:12.6;
x = -3.1:.1:5.6;
[X,Y] = meshgrid(x,y);
X = X(:);
Y = Y(:);

% convert data to arrays
Train = readmatrix('train.csv');
Test = readmatrix('test.csv');

% split data into classes
Test_1 = [];
Test_0 = [];
for i = 1:length(Test)
    if Test(i,3) == 1
        Test_1(end + 1, :) = Test(i, :);
    else
        Test_0(end + 1, :) = Test(i, :);
    end
end

Train_1 = [];
Train_0 = [];
for i = 1:length(Train)
    if Train(i,3) == 1
        Train_1(end + 1, :) = Train(i, :);
    else
        Train_0(end + 1, :) = Train(i, :);
    end
end

% estimate parameters
sigma_train_1 = cov(Train_1(:, 1:2));
mu_train_1 = mean(Train_1(:, 1:2));

sigma_train_0 = cov(Train_0(:,1:2));
mu_train_0 = mean(Train_0(:, 1:2));

classifier_b = [mvnpdf([X,Y], mu_train_0, sigma_train_0), mvnpdf([X,Y], mu_train_1, sigma_train_1)];

classified_b = [];
for i = 1: length(classifier_b)
    if classifier_b(i,1) > classifier_b(i,2)
        classified_b(i,:) = [X(i), Y(i), 0];
    else
        classified_b(i,:) = [X(i), Y(i), 1];
    end
end
% Plotting
classified_b_1 = [];
classified_b_0 = [];
for i = 1:length(classified_b)
    if classified_b(i,3) == 1
        classified_b_1(end + 1, :) = classified_b(i, 1:2);
    else
        classified_b_0(end + 1, :) = classified_b(i, 1:2);
    end
end

f1 = figure;
hold on
plot(classified_b_1(:,1), classified_b_1(:,2), 'ob')
plot(classified_b_0(:,1), classified_b_0(:,2), 'xr')
title('Bayesian Classifier')
legend('Class 1', 'Class 0')
saveas(f1, 'Bayesian Classification Boundary.jpg')

% Naive Bayesian Classifier

for i = 1:length(sigma_train_0(1,:))
    for j = 1: length(sigma_train_0(:,1))
        if i ~= j
            sigma_train_0(i,j) = 0;
        end
    end
end

for i = 1:length(sigma_train_1(1,:))
    for j = 1: length(sigma_train_1(:,1))
        if i ~= j
            sigma_train_1(i,j) = 0;
        end
    end
end

classifier_nb = [mvnpdf([X, Y], mu_train_0, sigma_train_0), mvnpdf([X,Y], mu_train_1, sigma_train_1)];

classified_nb = [];
for i = 1: length(classifier_nb)
    if classifier_nb(i,1) > classifier_nb(i,2)
        classified_nb(i,:) = [X(i), Y(i), 0];
    else
        classified_nb(i,:) = [X(i), Y(i), 1];
    end
end

% Plotting
classified_nb_1 = [];
classified_nb_0 = [];
for i = 1:length(classified_nb)
    if classified_nb(i,3) == 1
        classified_nb_1(end + 1, :) = classified_nb(i, 1:2);
    else
        classified_nb_0(end + 1, :) = classified_nb(i, 1:2);
    end
end
f2 = figure;
hold on
plot(classified_nb_1(:,1), classified_nb_1(:,2), 'ob')
plot(classified_nb_0(:,1), classified_nb_0(:,2), 'xr')
title('Naive Bayesian Classifier')
legend('Class 1', 'Class 0')
saveas(f2, 'Naive Bayesian Classification Boundary.jpg')

% Guassian Kernel
N1 = length(Train_1);
N0 = length(Train_0);
h = .5;
D = 2;
for i = 1: length(X)
    classifier_gk(i,1) = 0;
    classifier_gk(i,2) = 0;
    for j = 1: length(Train_0)
        classifier_gk(i,1) = classifier_gk(i,1) + 1 / (N0 * (2 * pi * h^2)^D/2) * exp(-norm([X(i),Y(i)]-Train_0(j,1:2))^2/(2*h^2));
    end
    for j = 1: length(Train_1)
        classifier_gk(i,2) = classifier_gk(i,2) + 1 / (N1 * (2 * pi * h^2)^D/2) * exp(-norm([X(i),Y(i)]-Train_1(j,1:2))^2/(2*h^2));
    end
end
for i = 1: length(classifier_gk)
    if classifier_gk(i,1) > classifier_gk(i,2)
        classified_gk(i,:) = [X(i), Y(i), 0];
    else
        classified_gk(i,:) = [X(i), Y(i), 1];
    end
end
% Plotting
classified_gk_1 = [];
classified_gk_0 = [];
for i = 1:length(classified_gk)
    if classified_gk(i,3) == 1
        classified_gk_1(end + 1, :) = classified_gk(i, 1:2);
    else
        classified_gk_0(end + 1, :) = classified_gk(i, 1:2);
    end
end

f3 = figure;
hold on
plot(classified_gk_1(:,1), classified_gk_1(:,2), 'ob')
plot(classified_gk_0(:,1), classified_gk_0(:,2), 'xr')
title('Gaussian Kernel')
legend('Class 1', 'Class 0')
saveas(f3, 'Gaussian Kernel Classification Boundary.jpg')

% k nearest neighbor
k = 9;
grid = [X, Y];
for j = 1:length(grid)
    for i = 1: length(Train)
        dist(j,i) = norm(Train(i,1:2)-grid(j,:));
    end
    [~,I] = mink(dist(j,:),k);
    if sum(Train(I,3)) > k/2
        classified_grid(j, :) = [grid(j, 1:2), 1];
    else
        classified_grid(j, :) = [grid(j, 1:2), 0];
    end
end
classified_grid_0 = [];
classified_grid_1 = [];
for i = 1: length(grid)
    if classified_grid(i, 3) == 1
        classified_grid_1(end+1, :) = classified_grid(i, 1:2);
    else
        classified_grid_0(end+1, :) = classified_grid(i, 1:2);
    end
end
f4 = figure;
hold on
plot(classified_grid_1(:,1), classified_grid_1(:,2), 'ob')
plot(classified_grid_0(:,1), classified_grid_0(:,2), 'xr')
title('K Nearest Neighbor')
legend('Class 1', 'Class 0')
saveas(f4, 'KNN Classification Boundary.jpg')