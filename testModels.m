function [score_models] = testModels(tLdata, tRdata, params)
%{
function takes in tLdata,tRdata, but will generate a default size matrix if
not given to the fxn, and outputs 4 models accuracy on the EEG signal
classification as a time series going back in time
 
-----------------------INPUT-------------------
tLdata (matrix):      total Right EEG data, with last column being a label
(trails x maxSamples+1) 

tRdata (matrix):      total Left EEG data, with last colummn being a label
(trails x maxSamples+1)
 
models (araray):      each model comes with a numeric label, in the
                      instance that not all want to be made, you can 
                      specify which ones with this array. Ex: [1 3 4]
                        label codes
                        knn: 1
                        cvknn: 2
                        logreg: 3
                        nvby: 4
                        SVM: 5
                        LDA: not yet coded

============PARAMS=================================================
window (int):      how large the sliding window is to be, default = 100
    
maxSamples (int):  max number of time samples to be aloud, default = 5,000 (columns)

trials (int):      number of trials to be generated, default = 10,000 (rows)
    
saveit (bool):       true: saves relevent ouput to .mat files,
                        scores.mat & lenstat.mat
                     false(default): doesn't save files

plotit (bool):       true(default): plots the models with acuracy vs time
                     false: doesn't display a plot

pass (bool):         true: does not perform model fitting, if plotit=true
                           then graph is generated before termination
                     false(default): performs model building
    
append (bool):       true: the specified number of trials will be generated AND
                         appended to a previous file version of tLdata.txt
                         & tRdata.txt
                     false(default): no data will be appended to file
-----------------------OUTPUT-------------------
knn_Correct (1 x nWins)array:      accuracy of a model trained on window
logreg_Correct (1 x nWins)array:   accuracy of a model trained on window
nvby_Correct (1 x nWins)array:     accuracy of a model trained on window
SVM_Correct (1 x nWins)array:      accuracy of a model trained on window
will display results in a graph, if plot is set to true. the default for plot is
true.
%}

%CHECK RELEVANT PARAMETERS EXISTENCE, ELSE SET TO DEFAULT
%SEE SUB-FUNCTIONS
if (~exist('params','var') || isempty(params)), params=[]; end
[window,models,maxSamples,trails,saveit,plotit,pass,append,gen,smoothin,params]...
    = GetParams (params);


% *  If the data is not in workspace of txt file, then generate data
% *  If you want to append data to an existing text file, (append = true),
% then will generate trails number of rows and append it to file
%^^ if maxSamples are different for saved data and current generated data,
%append will fail
% *  if none of the above, it will attempt to load from text file
if ((~exist('tLdata.txt','file') && isempty('tLdata','var')) || gen==true)
    disp('generating data...');
    [tLdata,tRdata,lens] = GenData(maxSamples,trails,false);
    fprintf('done \n \n');
elseif (append == true && exist('tLdata.txt','file'))
    disp('generating data...');
    [~,lens] = GenData(maxSamples,trails,append);
    fprintf('re-loading data...');
    load('tLdata.txt');
    load('tRdata.txt');
    fprintf('done \n \n');
elseif (exist('tLdata.txt','file') && isempty(tLdata))
    fprintf('loading data...');
    load('tLdata.txt');
    load('tRdata.txt');
    fprintf('done \n \n');
end
tRdata_c=tRdata;
tLdata_c=tLdata;
%global tLdata tRdata;

fprintf('__________TRAIL DURATION STATS____________  \n');
if (exist('lens','var'))
    std_length = std(lens);
    mean_length = mean(lens);
    mms=mean_length/1000;
    sms=std_length/1000;
    fprintf('Mean trail duration: %f  ',mean_length);
    fprintf('  or   %f sec \n', mms);
    fprintf('Standard deviation of trail duration: %f  ', std_length);
    fprintf('  or   %f sec \n \n \n',sms);
    if (saveit == true)
        fprintf('saving stats to file...');
        save('lenstat.mat','lens','std_length','mean_length');
        fprintf('variables [lens,std_length,mean_length] have been saved to file lenstat.mat \n \n \n');
    end
elseif (exist('lenstat.mat','file'))
    load lenstat.mat mean_length std_length;
    mms=mean_length/1000;
    sms=std_length/1000;
    fprintf('Mean trail duration: %f  ',mean_length);
    fprintf('  or   %f sec \n', mms);
    fprintf('Standard deviation of trail duration: %f  ', std_length);
    fprintf('  or   %f sec \n \n \n',sms);
else
    disp('could not find length statistics, try re-generating data');
end

%normalization, if need be
%tLdata_norm = (tLdata-min(min(tLdata)))/(max(max(tLdata))-min(min(tLdata)));
%tRdata_norm = (tRdata-min(min(tRdata)))/(max(max(tRdata))-min(min(tRdata)));

winsec=window/1000;
fprintf('______________________WINDOW SIZE_____________________ \n ');
fprintf('window size is %f time points -- or -- %f sec \n \n \n',window,winsec);

%PRINT DIMS
[row,col] = size(tLdata_c);
nWins = round((col-1)/(window/2)-1);
fprintf('__________DATA DIMENSIONS____________  \n');
fprintf('(rows)          trails in data: %d \n', row);
fprintf('(cols)    time samples in data: %d \n \n \n', (col-1));

%COMPARE THE NUMBER OF LEFT SUCESSES VS. RIGHT SUCCESSES
Rones = max(size(find(tLdata_c(:,col)==1)));
Lones = max(size(find(tRdata_c(:,col)==1)));
fprintf('___________LEFT VS. RIGHT LABELS__________ \n');
fprintf('number of Right successes(label=1): %d \n',Rones);
fprintf('number of Left successes(label=0): %d \n \n \n', Lones);
if(Rones ~= Lones)
    fprintf('Ooops, the labeled data isnt balanced. \n');
    fprintf('If its negligible, let models train. Otherwise, Ctrl+c to quit and try regenerating data \n \n');
end

str=strcat('accuracy_w',num2str(window));

num_models = max(size(models));
str=strcat(str,'_m',num2str(num_models));
score_labels = transpose(models);
score_models = [NaN(num_models,nWins) score_labels];
fprintf('Models Chosen: ');
if(any(models==1)), fprintf(' KNN '); end
if(any(models==2)), fprintf(' CV_KNN '); end
if(any(models==3)), fprintf(' Logistic Regression '); end
if(any(models==4)), fprintf(' Naive Bayes '); end
if(any(models==5)), fprintf(' SVM '); end
fprintf('\n \n \n');
input('PRESS ANY KEY TO CONTINUE, CTRL+C TO QUIT');
fprintf('\n \n \n');

dimerror=true;
if(exist('tLsmooth.txt','file') && smoothin==true)
    fprintf('loading saved smooth data...');
    load('tLsmooth.txt');
    load('tRsmooth.txt');
    fprintf(' done \n');
    [rowSm colSm] = size(tRsmooth);
    if((rowSm ~= row) || (colSm ~= col))
        fprintf('Oops, dimensions of smooth data and original data differ \n');
        fprintf('Data wil be re-smoothed and saved to match current working data \n');
    else
        tRdata_c=tRsmooth;
        tLdata_c=tLsmooth;
        dimerror=false;
    end
    str=strcat(str,'_sm');
end

if(smoothin==true && dimerror==true)
    tLsmooth=zeros(row,col);
    tRsmooth=zeros(row,col);
    for i = 1:row
        fprintf('smoothing trail: %d \n',i);
        tLsmooth(i,1:(col-1)) = transpose(smooth(tLdata_c(i,1:(col-1))));
        tRsmooth(i,1:(col-1)) = transpose(smooth(tRdata_c(i,1:(col-1))));
    end
    tLsmooth(:,col)=tLdata_c(:,col);
    tRsmooth(:,col)=tRdata_c(:,col) ;
    fprintf('smoothed data created and saving...');
    save('tLsmooth.txt','tLsmooth','-ascii');
    save('tRsmooth.txt','tRsmooth','-ascii');
    fprintf(' done \n');
    tRdata_c=tRsmooth;
    tLdata_c=tLsmooth;
    str=strcat(str,'_sm');
end

%SPLIT DATA
split = round(.8*row);
tL_train = tLdata_c(1:split,1:(col-1));
tL_test = tLdata_c((split+1):row,1:(col-1));
tR_train = tRdata_c(1:split,1:(col-1));
tR_test = tRdata_c((split+1):row,1:(col-1));


tR_train_labels = tRdata_c(1:split,col);
tL_train_labels = tLdata_c(1:split,col);
tR_test_labels = tRdata_c((split+1):row,col);
tL_test_labels = tLdata_c((split+1):row,col);

%USING RIGHT LABEL AS A DEFAULT, SO KNN ONLY PREDICTS BASED OFF OF ONE
%LABEL
y_train = tR_train_labels;
y_test = tR_test_labels;

%CREATE SLIDING WINDOW
col = col-1;
range_left = (col-window)+1;
range_right = col;

if (pass == true)
    if(plotit == true && exist('scores.mat','file'))
        load('scores','score_models');
        str=strcat(str,'.fig');
        plots(score_models,window,saveit,str);
    end
    return
end


fprintf('\n \n \n (WINDOW RANGE GIVEN IN SECONDS AFTER EVENT OCCURED) \n');
%ADJUST SLIDING WINDOW, FIT MODELS, AND SCORE MODELS
for i = 1:nWins
    sec_left = (col-range_left)/1000;
    sec_right = (col-range_right)/1000;
    fprintf('window range (sec)   : %f - %f \n',sec_left,sec_right);
    fprintf('preparing model group: %d / %d... \n',i,nWins);
    
    X_train_window = [ tL_train(:,range_left:range_right) tR_train(:,range_left:range_right) ];
    X_test_window = [ tL_test(:,range_left:range_right) tR_test(:,range_left:range_right) ];
    range_left = range_left-window/2;
    range_right = range_right-window/2;
    
    %CREATE & CROSS-VAL MODLELS
    X_all_window = [X_train_window ; X_test_window]; %for cv, must seperate data in function
    y_all = [y_train; y_test];
    %disp((size(X_all_window)));
    
    %=========================KNN=================================
    if(any(models==1))
        fprintf('working on KNN... ');
        model_knn = fitcknn(X_train_window,y_train,'NumNeighbors',30,'Standardize',1);
        [pred_knn,score_knn,cost] = predict(model_knn,X_test_window);
        [r,~]=find(score_models(:,(nWins+1))==1);
        score_models(r,i) = sum(y_test==pred_knn)/length(y_test);
        fprintf('done \n');
    end
    %=========================CVKNN=================================
    if(any(models==2))
        fprintf('working on CVKNN... ');
        model_cvknn = fitcknn(X_all_window,y_all,'NumNeighbors',30,'Standardize',1);
        cvmodel_knn = crossval(model_cvknn);  %10-fold crosval is default param  ,'Holdout',.8
        [pred_cvknn,score_cvknn,costcv] = kfoldPredict(cvmodel_knn);
        a=max(size(pred_cvknn));
        b=max(size(y_all));
        [r,~]=find(score_models(:,(nWins+1))==2);
        if(a ~= b)
            n=b-a;
            fprintf('   ..uneven dims..  ');
            pred_cvknn=[zeros(n,1); pred_cvknn];
            score_models(r,i) = sum(y_all==pred_cvknn)/length(y_all);
        else
            score_models(r,i) = sum(y_all==pred_cvknn)/length(y_all);
        end
        fprintf('done \n');
    end 
    %=========================LOGREG=================================
    if(any(models==3))
        fprintf('working on LogReg... ');
        opts=statset('MaxIter',10000,'UseParallel',true);
        model_logreg = fitglm(X_train_window,y_train,'Distribution','binomial','Link','logit','Options',opts);
        pred_logreg = predict(model_logreg,X_test_window);
        [r,~]=find(score_models(:,(nWins+1))==3);
        score_models(r,i) = mean(y_test==round(pred_logreg));
        fprintf('done \n');
    end
    %=========================NAIVE BAYES=================================
    if(any(models==4))
        fprintf('working on Naive Bayes... ');
        model_nvby = fitcnb(X_train_window,y_train);
        pred_nvby = predict(model_nvby,X_test_window);
        [r,~]=find(score_models(:,(nWins+1))==4);
        score_models(r,i) = sum(y_test==pred_nvby)/length(y_test);
        fprintf('done \n');
    end
    %=========================SVM=================================
    if(any(models==5))
        fprintf('working on SVM... ');
        model_SVM = fitcsvm(X_train_window,y_train,'Standardize',true,'KernelFunction','linear');
        pred_SVM = predict(model_SVM,X_test_window);
        [r,~]=find(score_models(:,(nWins+1))==2);
        score_models(r,i) = sum(y_test==pred_SVM)/length(y_test);
        fprintf('done \n');
    end 
    fprintf('done \n \n');
end

if (plotit == true)
    plots(score_models,window,saveit,str)
end

if (saveit == true || ~exist('scores.mat','file'))
    fprintf('saving model scores to file...');
    save('scores.mat','score_models');
    fprintf('done, saved to scores.mat \n \n');
end

end

%======================== Internal subfunctions ==========================
function [window,models,maxSamples,trails,saveit,plotit,pass,append,gen,smoothin,params]...
    = GetParams (params)
% If the parameters are fields of 'params', extract them. Otherwise, give
% them their default values.

%size of sliding widow for training models
%defalut = 100
if (isfield(params,'window'))
    window=params.window;
else
    window = 100; 
    params.window=window;
end

%which models to run
%defalut = all
if (isfield(params,'models'))
    models=params.models;
else
    models = [1 2 3 4 5]; 
    params.models=models;
end

%number of time points to be taken in the data, corresponds to col
%default = 5,000  or 5sec
if (isfield(params,'maxSamples'))
    maxSamples=params.maxSamples;
else
    maxSamples = 5000; 
    params.maxSamples=maxSamples;
end

%number of trials to be generated, corresponds to rows in the data
%default = 10,0000
if (isfield(params,'trails'))
    trails=params.trails;
else
    trails = 10000; 
    params.trails=trails;
end

%whether or not to save the accuracy scores of the models / duration
%statistics
%default = false
if (isfield(params,'saveit'))
    saveit=params.saveit;
else
    saveit = false; 
    params.saveit=saveit;
end

%whether or not the accuracy vs time plot is generated
%default = true
if (isfield(params,'plotit'))
    plotit=params.plotit; 
else
    plotit = true;
    params.plotit=plotit;
end

%whether or not you ant to pas the model building, in case you want to only
%plot
%default = false
if (isfield(params,'pass'))
    pass=params.pass; 
else
    pass=false;
    params.pass=pass;
end

%whether or not you want to generate more data and append it to the data 
%txt file, NOTE: maxSamples, or columns, need to match that of the tLdata
%                txt file, other wise there wil be an error
%default = false
if (isfield(params,'append'))
    append=params.append; 
else
    append=false;
    params.append=append;
end

%whether or not you want to generate data, 
%default = false
if (isfield(params,'gen'))
    gen=params.gen; 
else
    gen=false;
    params.gen=gen;
end

if (isfield(params,'smoothin'))
    smoothin=params.smoothin; 
else
    smoothin=false;
    params.smoothin=smoothin;
end

end




%{
beta = .15;
params.c=.05; 
[~,l,r]=myModel([.04,.05], [.5,.6],[beta beta],[],params);
t=(0:length(l)-1)./1000;

ls=transpose(smooth(l));
rs=transpose(smooth(r));

subplot(2,2,1), plot(t,l); 
subplot(2,2,2), plot(t,r);
subplot(2,2,3), plot(t,ls); 
subplot(2,2,4), plot(t,rs);
%}

%{

inputSize = 100;
numResponses = 1;
numHiddenUnits = 200;
outputSize = 2;

layers = [ ...
    sequenceInputLayer(inputSize)
    lstmLayer(numHiddenUnits)
    fullyConnectedLayer(numResponses)
    regressionLayer];

opts = trainingOptions('adam', ...
    'MaxEpochs',250, ...
    'GradientThreshold',1, ...
    'InitialLearnRate',0.005, ...
    'LearnRateSchedule','piecewise', ...
    'LearnRateDropPeriod',125, ...
    'LearnRateDropFactor',0.2, ...
    'Verbose',0, ...
    'Plots','training-progress');

net = trainNetwork(XTrain,YTrain,layers,opts);
%}

 %{
https://www.mathworks.com/help/stats/
https://www.mathworks.com/help/nnet/ug/deep-learning-in-matlab.html
https://www.mathworks.com/help/nnet/examples/time-series-forecasting-using-deep-learning.html
https://www.mathworks.com/help/nnet/gs/neural-network-time-series-prediction-and-modeling.html
https://www.mathworks.com/help/nnet/examples/classify-sequence-data-using-lstm-networks.html
https://www.mathworks.com/help/nnet/ref/network.html
%}