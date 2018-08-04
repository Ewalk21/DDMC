function plots(score_models,window,saveit,str)
    %{
    model label codes
    knn_Correct: 1
    cvknn_Correct: 2
    logreg_Correct: 3
    nvby_Correct: 4
    SVM_Correct: 5
    LDA_Correct: not yet coded
    %}
    if(~exist('str','var')),str='accuracy.fig';end

    nWins = max(size(score_models))-1; %-1 for labels
    
    labels=[];
    [rows cols] = size(score_models);
    
    totaltime = (window/2)*nWins/1000;
    dt = (window/2)/1000;
    t = 0:dt:(totaltime-dt);
    tick = 0:dt:(totaltime-dt);
    tick = correctTicks(tick,dt,dt,totaltime);
    
    fig=figure;
    ylim([0 1]);
    plot(t,repelem(.5,nWins),'-r','Linewidth',.8);
    hold on;
    labels=[labels , "50% acc. line"];
    for i=1:rows
        %if its mostly nans, pass it, on to the next
        a = max(size(find(isnan(score_models(i,1:(cols-1))) == 1)));
        if(a > .5*cols), continue; end
        
        model_label = score_models(i,cols);
        labels = labelChecker(model_label,labels);
        
        plot(t,score_models(i,1:(cols-1)),'.-','MarkerSize',4,'Linewidth',.9);
        hold on;
    end
    xticks(tick); 
    %ylim([0 1]);
    ylabel('Accuracy');
    xlabel('Time (Seconds)');
    title(['Accuracy VS. Time (sec)    (window size = ', num2str(window), ' or ', num2str(window/1000), 'sec)']);
    lgd = legend(labels,'Location','northeast');
    title(lgd,'Models');
    %dim = [0.3 0.1 0.3 0.5];
    %annotation('textbox',dim,'String',str,'FitBoxToText','on');
    if (saveit==true)
        saveas(fig,str);
    end
end

%recursive function to generate a reasonable number of ticklabels
%no more than 20 tickLabels
function tickNew = correctTicks(tick,dt,endpnt,totaltime)
    if(max(size(tick)) < 20)
        tickNew = tick;
        return
    else
        tick = 0:dt:(totaltime-endpnt);
    end
    tickNew = correctTicks(tick,2*dt,endpnt,totaltime);
end

function labelsnew = labelChecker(model_label,labels)
    if(model_label == 1)
        labelsnew=[labels , "KNN"];
    elseif(model_label == 2)
        labelsnew=[labels , "CVKNN"];
    elseif(model_label == 3)
        labelsnew=[labels , "Logistic Regression"];
    elseif(model_label == 4)
        labelsnew=[labels , "Naive Bayes"];
    elseif(model_label == 5)
        labelsnew=[labels , "SVM"];
    else
        labelsnew=[labels , "unknownModel"];
    end
end







