clear
Train = readmatrix('train.csv');
Test = readmatrix('test.csv');

Train_1 = [];
Train_0 = [];
for i = 1:length(Train)
    if Train(i,3) == 1
        Train_1(end + 1, :) = Train(i, :);
    else
        Train_0(end + 1, :) = Train(i, :);
    end
end
f1 = figure;
plot(Train_1(:, 1), Train_1(:, 2), 'ob')
hold on
plot(Train_0(:, 1), Train_0(:, 2), 'xr')
title('Training Data')
legend('Class 1', 'Class 0')
saveas(f1, 'Training Data.jpg')

Test_1 = [];
Test_0 = [];
for i = 1:length(Test)
    if Test(i,3) == 1
        Test_1(end + 1, :) = Test(i, :);
    else
        Test_0(end + 1, :) = Test(i, :);
    end
end
f2 = figure;
plot(Test_1(:, 1), Test_1(:, 2), 'ob')
hold on
plot(Test_0(:, 1), Test_0(:, 2), 'xr')
title('Testing Data')
legend('Class 1', 'Class 0')
saveas(f2, 'Testing Data.jpg')