classdef NeuralNetwork < handle
    properties (Access = private)
        lr(1, 1) double {mustBeNumeric}
        wih(:,:) double {mustBeNumeric}
        who(:,:) double {mustBeNumeric}
    end
    methods (Access = public)
        function obj = NeuralNetwork(varargin)
            if nargin == 4
                obj.lr = varargin{4};
                obj.wih = normrnd(0,1/sqrt(varargin{1}), varargin{2}, varargin{1});
                obj.who = normrnd(0,1/sqrt(varargin{2}), varargin{3}, varargin{2});
            elseif nargin == 1
                obj.open_configuration(varargin{1});
            else
                error('Incorrect number of arguments.');
            end
        end
        function outputs = query(obj, inputs)
            hidden_inputs = obj.wih * inputs;
            hidden_outputs = obj.activation_function(hidden_inputs);
            final_inputs = obj.who * hidden_outputs;
            outputs = obj.activation_function(final_inputs);
        end
        function train(obj, inputs, targets)
            hidden_inputs = obj.wih * inputs;
            hidden_outputs = obj.activation_function(hidden_inputs);
            final_inputs = obj.who * hidden_outputs;
            final_outputs = obj.activation_function(final_inputs);
            output_errors = targets - final_outputs;
            hidden_errors = obj.who' * output_errors;
            obj.who = obj.who + (obj.lr * output_errors .* final_outputs .* (1 - final_outputs)) * hidden_outputs';
            obj.wih = obj.wih + (obj.lr * hidden_errors .* hidden_outputs .* (1 - hidden_outputs)) * inputs';
        end 
        function save_configuration(obj,filename)
            learning_rate = obj.lr;
            weights_input_hidden = obj.wih;
            weights_hidden_output = obj.who;
            save(filename, 'learning_rate', 'weights_input_hidden', 'weights_hidden_output');
        end
        function open_configuration(obj,filename)
            load(filename, 'learning_rate', 'weights_input_hidden', 'weights_hidden_output');
            obj.lr = learning_rate;
            obj.wih = weights_input_hidden;
            obj.who = weights_hidden_output;
        end
    end
    methods (Access = private)
        function outputsignal = activation_function(~, inputsignal)
            outputsignal = 1 ./ (1 + exp(-inputsignal));
        end
    end
end
