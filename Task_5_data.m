clear
% convert data to arrays
Train = readmatrix('train.csv');
Test = readmatrix('test.csv');

Train_1 = [];
Train_0 = [];
for i = 1:length(Train)
    if Train(i,3) == 1
        Train_1(end + 1, :) = Train(i, 1:2);
    else
        Train_0(end + 1, :) = Train(i, 1:2);
    end
end
n = 0;
for k = 1:2:15
    
    n = n + 1;
    for j = 1:length(Train)
        for i = 1: length(Train)
            dist(j,i) = norm(Train(i,1:2)-Train(j,1:2));
        end
        [~,I] = mink(dist(j,:),k);
        if sum(Train(I,3)) > k/2
            classified_train(j, :) = [Train(j, 1:2), 1];
        else
            classified_train(j, :) = [Train(j, 1:2), 0];
        end
    end
    
    % accuracy
    correct_train = 0;
    for i = 1: length(Train)
        if classified_train(i, 3) == Train(i, 3)
            correct_train = correct_train + 1;
        end
    end
    train_accuracy(n) = correct_train / length(Train);
    
    for j = 1:length(Test)
        for i = 1: length(Train)
            dist(j,i) = norm(Train(i,1:2)-Test(j,1:2));
        end
        [~,I] = mink(dist(j,:),k);
        if sum(Train(I,3)) > k/2
            classified_test(j, :) = [Test(j, 1:2), 1];
        else
            classified_test(j, :) = [Test(j, 1:2), 0];
        end
    end
    
    % test accuracy
    correct_test = 0;
    for i = 1: length(Test)
        if Test(i,3) == classified_test(i,3)
            correct_test = correct_test + 1;
        end
    end
    
    test_accuracy(n) = correct_test / length(Test);
end
k = 1:2:15;
[max_accuracy, I] = max(test_accuracy);
fprintf('The maximum testing accuracy for the K Nearest Neighbor method is %2.2f%% and occurs at k = %d\n', max_accuracy*100, k(I))
fprintf('The training accuracy at k = %d is %2.2f%%\n', k(I), train_accuracy(I)*100)

k = k(I);
for j = 1:length(Train)
    for i = 1: length(Train)
        dist(j,i) = norm(Train(i,1:2)-Train(j,1:2));
    end
    [~,I] = mink(dist(j,:),k);
    if sum(Train(I,3)) > k/2
        classified_train(j, :) = [Train(j, 1:2), 1];
    else
        classified_train(j, :) = [Train(j, 1:2), 0];
    end
end

for j = 1:length(Test)
    for i = 1: length(Train)
        dist(j,i) = norm(Train(i,1:2)-Test(j,1:2));
    end
    [~,I] = mink(dist(j,:),k);
    if sum(Train(I,3)) > k/2
        classified_test(j, :) = [Test(j, 1:2), 1];
    else
        classified_test(j, :) = [Test(j, 1:2), 0];
    end
end

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
title('Classified Training Data - KNN')
legend('Class 1', 'Class 0')
saveas(f1, 'KNN Classification of Training Data.jpg')

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
title('Classified Testing Data - KNN')
legend('Class 1', 'Class 0')
saveas(f2, 'KNN Classification of Testing Data.jpg')