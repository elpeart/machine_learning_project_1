clear
train = readmatrix('zipcode_train.csv');
test = readmatrix('zipcode_test.csv');

train_3d = zeros(300,16,10);
index = ones(1,10);
for i = 1: length(train)
    for j = 1:10
        if train(i,17) == j
            train_3d(index(j), :, j) = train(i, 1:16);
            index(j) = index(j)+1;
        end
    end
end
N = length(train_3d(:,1,1));
D = length(train_3d(1,:,1));
n = 0;
% h = 0.5;
for h = .3:.2:.7
    for i = 1: length(train)
        classifier(i, :) = zeros(1, length(train_3d(1, :, 1)));
        for j = 1: length(train_3d(1,1,:))
            for k = 1: length(train_3d(:, 1, 1))
                classifier(i, j) = classifier(i,j) + 1 / (N * (2 * pi * h^2)^D/2) * exp(-norm(train(i,1:16)-train_3d(k, :, j))^2/(2*h^2));
            end
        end
        [~, class] = max(classifier(i,:));
        classified_train(i,:) = [train(i, 1:16), class];
    end
    
    % Training accuracy
    correct_train = 0;
    for i = 1: length(train)
        if train(i,17) == classified_train(i,17)
            correct_train = correct_train + 1;
        end
    end
    n = n + 1;
    train_accuracy(n) = correct_train / length(train);
    
    for i = 1: length(test)
        classifier(i, :) = zeros(1, length(train_3d(1, :, 1)));
        for j = 1: length(train_3d(1,1,:))
            for k = 1: length(train_3d(:, 1, 1))
                classifier(i, j) = classifier(i,j) + 1 / (N * (2 * pi * h^2)^D/2) * exp(-norm(test(i,1:16)-train_3d(k, :, j))^2/(2*h^2));
            end
        end
        [~, class] = max(classifier(i,:));
        classified_test(i,:) = [train(i, 1:16), class];
    end
    
    correct_test = 0;
    for i = 1: length(train)
        if test(i,17) == classified_test(i,17)
            correct_test = correct_test + 1;
        end
    end
    test_accuracy(n) = correct_test / length(test);
end
[max_accuracy, I] = max(test_accuracy);
h1 = .3:.2:.7;
fprintf('The maximum testing accuracy occurs at h = %1.2f\n', h1(I))
fprintf('The training accuracy at h = %1.2f is %2.2f%%\n', h1(I), train_accuracy(I)*100)
fprintf('The testing accuracy at h = %1.2f is %2.2f%%\n', h1(I), test_accuracy(I)*100)