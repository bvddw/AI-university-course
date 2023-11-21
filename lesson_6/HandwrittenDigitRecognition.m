function HandwrittenDigitRecognition
    fig = figure('Units','pixels','Position',[450, 250, 1150, 700],...
        'NumberTitle','off','Name','HandwrittenDigitRecognition','MenuBar','none',...
        'Resize','off','WindowButtonDownFcn',@fig_WindowButtonDownFcn);
    ax = axes('Units','pixels','Position',[50, 150, 500, 500],...
        'Box','on','XTick',[],'YTick',[]);
    axis('manual')
    hold(ax,'on')
    result = uicontrol('Style','edit','Units','pixels','Position',[600, 150, 500, 500],...
        'String','','FontSize',300,'Enable','off');
    uicontrol('Style','pushbutton','Units','pixels',...
        'Position',[50, 38, 500, 80],'String','Recognize','FontSize',38,...
        'FontName','BankGothic Md BT','Callback',@btn_Callback);
    uicontrol('Style','pushbutton','Units','pixels',...
        'Position',[600, 38, 300, 80],'String','Train','FontSize',20,...
        'FontName','BankGothic Md BT','Callback',@train_Callback);
    uicontrol('Style','edit','Units','pixels',...
        'Position',[910, 38, 190, 80],'String','','FontSize',20,...
        'FontName','Helvetica','Callback',@cor_answer_Callback);
    
    nnet = NeuralNetwork('config.mat');
    cor_answer_value = '';

    function fig_WindowButtonDownFcn(~,~)
        if strcmp(get(fig, 'SelectionType'), 'normal')
            set(fig, 'WindowButtonMotionFcn', @fig_WindowButtonMotionFcn)
            set(fig, 'WindowButtonUpFcn', @fig_WindowButtonUpFcn)
        elseif strcmp(get(fig,'SelectionType'),'alt')
            cla(ax)
            set(result, 'String', '');
        end
    end

    function fig_WindowButtonMotionFcn(~,~)
        cp = get(ax,'CurrentPoint');
        plot(cp(1,1),cp(1,2),'Marker','O','MarkerSize',30,...
            'MarkerEdgeColor','k','MarkerFaceColor','k')
    end

    function fig_WindowButtonUpFcn(~,~)
        set(fig,'WindowButtonMotionFcn',[])
        set(fig,'WindowButtonUpFcn',[])
    end

    function btn_Callback(~,~)
        img = rgb2gray(frame2im(getframe(ax,[1 1 498 498])));
        img = imresize(img, [28 28]);
        inputs = double(reshape(fliplr(rot90(img,-1)),[],1));
        inputs = (255.0-inputs) / 255.0 * 0.99 + 0.01;
        outputs = nnet.query(inputs);
        [~,I] = max(outputs);
        set(result, 'String', num2str(I - 1));
    end

    function train_Callback(~,~)
        img = rgb2gray(frame2im(getframe(ax,[1 1 498 498])));
        img = imresize(img, [28 28]);
        inputs = double(reshape(fliplr(rot90(img,-1)),[],1));
        inputs = (255.0-inputs) / 255.0 * 0.99 + 0.01;
        cor_answer_numeric = round(str2double(cor_answer_value));
        disp(cor_answer_numeric)
        if ~isnan(cor_answer_numeric) && ismember(cor_answer_numeric, 0:9)
            targets = zeros(10, 1) + 0.01;
            targets(cor_answer_numeric + 1) = 0.99;
            nnet.train(inputs, targets);
            cla(ax);
            set(result, 'String', '');
        else
            disp('Invalid input for the correct answer. Please enter a number between 0 and 9.');
        end
    end

    function cor_answer_Callback(src, ~)
        cor_answer_value = get(src, 'String');
    end
end
