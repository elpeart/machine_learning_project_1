clear
train = readmatrix('zipcode_train.csv');
test = readmatrix('zipcode_test.csv');
n = 0;
for k = 1:4:13
    for j = 1:length(train)
        for i = 1: length(train)
            dist(j,i) = norm(train(i,1:16)-train(j,1:16));
        end
        [~,I] = mink(dist(j,:),k);
        class = mode(train(I, 17));
        classified_train(j, :) = [train(j, 1:16), class];
    end
    
    n = n + 1;
    correct_train = 0;
    for i = 1: length(train)
        if classified_train(i, 17) == train(i, 17)
            correct_train = correct_train + 1;
        end
    end
    train_accuracy(n) = correct_train / length(train);
    
    for j = 1:length(test)
        for i = 1: length(train)
            dist(j,i) = norm(train(i,1:16)-test(j,1:16));
        end
        [~,I] = mink(dist(j,:),k);
        class = mode(train(I, 17));
        classified_test(j, :) = [test(j, 1:16), class];
    end
    
    correct_test = 0;
    for i = 1: length(test)
        if classified_test(i, 17) == test(i, 17)
            correct_test = correct_test + 1;
        end
    end
    test_accuracy(n) = correct_test / length(test);
end
[max_accuracy, I] = max(test_accuracy);
k = 1:4:13;
fprintf('The maximum testing accuracy for the K Nearest Neighbor method is %2.2f%% and occurs at k = %d\n', max_accuracy*100, k(I))
fprintf('The training accuracy at k = %d is %2.2f%%\n', k(I), train_accuracy(I)*100)