function digitRecognizer
    input_nodes = 784;
    hidden_nodes = 200;
    output_nodes = 10;

    learning_rate = 0.1;

    nnet = NeuralNetwork(input_nodes, hidden_nodes, output_nodes, learning_rate);
    nnet = train_fcn(nnet);
    test_fcn(nnet);
    nnet.save_configuration('config.mat');

    function nnet = train_fcn(nnet)
        disp('Neural netwrok training...');
        train_data = readmatrix('mnist_train.csv');
        epochs = 5;
        for e = 1:epochs
            disp(['Epoch ', num2str(e)])
            for i = 1:size(train_data, 1)
                inputs = ((train_data(i, 2:end) / 255.0 * 0.99) + 0.01)';
                targets = zeros(output_nodes, 1) + 0.01;
                targets(train_data(i, 1) + 1) = 0.99;
                nnet.train(inputs, targets);
            end 
        end 
    end
    
    function test_fcn(nnet)
        disp('Neural network testing...');
        score = 0;
        test_data = readmatrix('mnist_test.csv');
        for i = 1:size(test_data, 1)
            correct_label = test_data(i, 1);
            inputs = ((test_data(i, 2:end) / 255.0 * 0.99) + 0.01)';
            outputs = nnet.query(inputs);
            [~, I] = max(outputs);
            label = I - 1;
            if label == correct_label
                score = score + 1;
            end 
        end 
        disp(['Efficiency: ', num2str(score / size(test_data, 1) * 100), ' %'])
    end
end
