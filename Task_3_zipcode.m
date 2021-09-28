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

mu = mean(train_3d);

for i = 1:10
    sigma(:,:,i) = cov(train_3d(:,:,i));
    d(i) = det(sigma(:,:,i));
    if d(i) == 0
        sigma(:,:,i) = sigma(:,:,i) + eye(16) * .03;
    end
end

for k = 1: length(sigma(1,1,:))
    for i = 1:length(sigma(1,:,1))
        for j = 1: length(sigma(:,1,1))
            if i ~= j
                sigma(i,j,k) = 0;
            end
        end
    end
end

for j = 1:10
    classifier_train(:,j) = 1/10 * mvnpdf(train(:,1:16), mu(:,:,j), sigma(:,:,j));
end

for i = 1: length(classifier_train)
    [~,I(i)] = max(classifier_train(i,:));
end

for i = 1: length(train)
    classified_train(i,:) = [train(i,1:16), I(i)];
end
% training accuracy
correct_train = 0;
for i = 1: length(train)
    if classified_train(i,17) == train(i,17)
        correct_train = correct_train + 1;
    end
end
train_accuracy = correct_train / length(train);
fprintf('Training accuracy for this naive Bayesian classifier on the zip code data is %2.2f%%\n',train_accuracy*100)

for j = 1:10
    classifier_test(:,j) = 1/10 * mvnpdf(test(:,1:16), mu(:,:,j), sigma(:,:,j));
end

for i = 1: length(classifier_test)
    [~,I(i)] = max(classifier_test(i,:));
end

for i = 1: length(test)
    classified_test(i,:) = [test(i,1:16), I(i)];
end
% testing accuracy
correct_test = 0;
for i = 1: length(test)
    if classified_test(i,17) == test(i,17)
        correct_test = correct_test + 1;
    end
end
test_accuracy = correct_test / length(test);
fprintf('Testing accuracy for this naive Bayesian classifier on the zip code data is %2.2f%%\n',test_accuracy*100)