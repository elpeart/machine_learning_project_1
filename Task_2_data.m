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

classifier_train = [mvnpdf(Train(:,1:2), mu_train_0, sigma_train_0) * 0.5, mvnpdf(Train(:,1:2), mu_train_1, sigma_train_1) * 0.5];

classified_train_data = [];
for i = 1: length(classifier_train)
    if classifier_train(i,1) > classifier_train(i,2)
        classified_train_data(i,:) = [Train(i,1:2), 0];
    else
        classified_train_data(i,:) = [Train(i,1:2), 1];
    end
end

% accuracy
correct_train = 0;
for i = 1: length(Train)
    if Train(i,3) == classified_train_data(i,3)
        correct_train = correct_train + 1;
    end
end
train_accuracy = correct_train / length(Train);
fprintf('Training accuracy for this Bayesian classifier on the generated data is %2.2f%%\n',train_accuracy*100)

% Plotting
classified_train_1 = [];
classified_train_0 = [];
for i = 1:length(Train)
    if classified_train_data(i,3) == 1
        classified_train_1(end + 1, :) = classified_train_data(i, 1:2);
    else
        classified_train_0(end + 1, :) = classified_train_data(i, 1:2);
    end
end

f1 = figure;
hold on
plot(classified_train_1(:,1), classified_train_1(:,2), 'ob')
plot(classified_train_0(:,1), classified_train_0(:,2), 'xr')
title('Classified Training Data - Bayesian')
legend('Class 1', 'Class 0')
saveas(f1, 'Bayesian Classification of Training Data.jpg')

classifier_test = [mvnpdf(Test(:,1:2), mu_train_0, sigma_train_0) * 0.5, mvnpdf(Test(:,1:2), mu_train_1, sigma_train_1) * 0.5];

classified_test_data = [];
for i = 1: length(classifier_test)
    if classifier_test(i,1) > classifier_test(i,2)
        classified_test_data(i,:) = [Test(i,1:2), 0];
    else
        classified_test_data(i,:) = [Test(i,1:2), 1];
    end
end

% accuracy
correct_test = 0;
for i = 1: length(Test)
    if Test(i,3) == classified_test_data(i,3)
        correct_test = correct_test + 1;
    end
end
test_accuracy = correct_test / length(Test);
fprintf('Testing accuracy for this Bayesian classifier on the generated data is %2.2f%%\n',test_accuracy*100)


classified_test_1 = [];
classified_test_0 = [];
for i = 1:length(Test)
    if classified_test_data(i,3) == 1
        classified_test_1(end + 1, :) = classified_test_data(i, 1:2);
    else
        classified_test_0(end + 1, :) = classified_test_data(i, 1:2);
    end
end

f2 = figure;
hold on
plot(classified_test_1(:,1), classified_test_1(:,2), 'ob')
plot(classified_test_0(:,1), classified_test_0(:,2), 'xr')
title('Classified Testing Data - Bayesian')
legend('Class 1', 'Class 0')
saveas(f2, 'Bayesian Classification of Testing Data.jpg')