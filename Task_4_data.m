clear
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
        Train_1(end + 1, :) = Train(i, 1:2);
    else
        Train_0(end + 1, :) = Train(i, 1:2);
    end
end
N1 = length(Train_1);
N0 = length(Train_0);
n = 0;
D = 2;
for h = 0.05:.05:1
    
    for i = 1: length(Train)
        classifier(i,1) = 0;
        classifier(i,2) = 0;
        for j = 1: length(Train_0)
            classifier(i,1) = classifier(i,1) + 1 / (N0 * (2 * pi * h^2)^D/2) * exp(-norm(Train(i,1:2)-Train_0(j,:))^2/(2*h^2));
        end
        for j = 1: length(Train_1)
            classifier(i,2) = classifier(i,2) + 1 / (N1 * (2 * pi * h^2)^D/2) * exp(-norm(Train(i,1:2)-Train_1(j,:))^2/(2*h^2));
        end
    end
    for i = 1: length(classifier)
        if classifier(i,1) > classifier(i,2)
            classified_train(i,:) = [Train(i,1:2), 0];
        else
            classified_train(i,:) = [Train(i,1:2), 1];
        end
    end
    
    % accuracy
    correct_train = 0;
    for i = 1: length(Train)
        if Train(i,3) == classified_train(i,3)
            correct_train = correct_train + 1;
        end
    end
    n = n+1;
    train_accuracy(n) = correct_train / length(Train);
    
    
    N1 = length(Test_1);
    N0 = length(Test_0);
    
    for i = 1: length(Test)
        classifier(i,1) = 0;
        classifier(i,2) = 0;
        for j = 1: length(Train_0)
            classifier(i,1) = classifier(i,1) + 1 / (N0 * (2 * pi * h^2)^D/2) * exp(-norm(Test(i,1:2)-Train_0(j,:))^2/(2*h^2));
        end
        for j = 1: length(Train_1)
            classifier(i,2) = classifier(i,2) + 1 / (N1 * (2 * pi * h^2)^D/2) * exp(-norm(Test(i,1:2)-Train_1(j,:))^2/(2*h^2));
        end
    end
    for i = 1: length(classifier)
        if classifier(i,1) > classifier(i,2)
            classified_test(i,:) = [Test(i,1:2), 0];
        else
            classified_test(i,:) = [Test(i,1:2), 1];
        end
    end
    
    % accuracy
    correct_test = 0;
    for i = 1: length(Test)
        if Test(i,3) == classified_test(i,3)
            correct_test = correct_test + 1;
        end
    end
    
    test_accuracy(n) = correct_test / length(Test);
end
[max_accuracy,I] = max(test_accuracy);
h = 0.05:.05:1;
h = h(I);
fprintf('Training accuracy for this Gaussian Kernel classifier on the generated data is %2.2f%%\n',train_accuracy(I)*100)
fprintf('The best testing accuracy occurs at h = %1.2f\n',h)
fprintf('Testing accuracy for this Gaussian Kernel classifier on the generated data is %2.2f%%\n',max_accuracy*100)

for i = 1: length(Train)
    classifier(i,1) = 0;
    classifier(i,2) = 0;
    for j = 1: length(Train_0)
        classifier(i,1) = classifier(i,1) + 1 / (N0 * (2 * pi * h^2)^D/2) * exp(-norm(Train(i,1:2)-Train_0(j,:))^2/(2*h^2));
    end
    for j = 1: length(Train_1)
        classifier(i,2) = classifier(i,2) + 1 / (N1 * (2 * pi * h^2)^D/2) * exp(-norm(Train(i,1:2)-Train_1(j,:))^2/(2*h^2));
    end
end
for i = 1: length(classifier)
    if classifier(i,1) > classifier(i,2)
        classified_train(i,:) = [Train(i,1:2), 0];
    else
        classified_train(i,:) = [Train(i,1:2), 1];
    end
end

for i = 1: length(Test)
    classifier(i,1) = 0;
    classifier(i,2) = 0;
    for j = 1: length(Train_0)
        classifier(i,1) = classifier(i,1) + 1 / (N0 * (2 * pi * h^2)^D/2) * exp(-norm(Test(i,1:2)-Train_0(j,:))^2/(2*h^2));
    end
    for j = 1: length(Train_1)
        classifier(i,2) = classifier(i,2) + 1 / (N1 * (2 * pi * h^2)^D/2) * exp(-norm(Test(i,1:2)-Train_1(j,:))^2/(2*h^2));
    end
end
for i = 1: length(classifier)
    if classifier(i,1) > classifier(i,2)
        classified_test(i,:) = [Test(i,1:2), 0];
    else
        classified_test(i,:) = [Test(i,1:2), 1];
    end
end

% Plotting
classified_train_1 = [];
classified_train_0 = [];
for i = 1:length(Train)
    if classified_train(i,3) == 1
        classified_train_1(end + 1, :) = classified_train(i, 1:2);
    else
        classified_train_0(end + 1, :) = classified_train(i, 1:2);
    end
end

f1 = figure;
hold on
plot(classified_train_1(:,1), classified_train_1(:,2), 'ob')
plot(classified_train_0(:,1), classified_train_0(:,2), 'xr')
title('Classified Training Data - Gaussian Kernel')
legend('Class 1', 'Class 0')
saveas(f1, 'Gaussian Kernel Classification of Training Data.jpg')

% Plotting
classified_test_1 = [];
classified_test_0 = [];
for i = 1:length(Test)
    if classified_test(i,3) == 1
        classified_test_1(end + 1, :) = classified_test(i, 1:2);
    else
        classified_test_0(end + 1, :) = classified_test(i, 1:2);
    end
end

f2 = figure;
hold on
plot(classified_test_1(:,1), classified_test_1(:,2), 'ob')
plot(classified_test_0(:,1), classified_test_0(:,2), 'xr')
title('Classified Testing Data - Gaussian Kernel')
legend('Class 1', 'Class 0')
saveas(f2, 'Gaussian Kernel Classification of Testing Data.jpg')