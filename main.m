%{
testModels will run all of the models with the specified window size.

if the input data is empty, txt file doesnt exist, or gen == true, 
then testModels will generate data specified by maxsamples and trials. 

if append was true and txt fileexists, 
then testModels will generate data specified by maxsamples and trials
and append it to the existing txt file while also updating duration
statistics.

params holds relevant parameters for testing models, default values are
assigned if they are not passed in.


params.models code labels
knn_Correct:      1
cvknn_Correct:    2
logreg_Correct:   3
nvby_Correct:     4
SVM_Correct:      5
LDA_Correct: not yet coded
%}
windows = [25 50 75 100 150 200 300 400];
params = [];

params.window=50;           %size of sliding window
                                 %default:100

params.models=[3];           %which models to run
                                 %default:all

params.maxSamples=5000;     %number of cols to be gen
                                 %default:5,000

params.trails=5000;         %number of rows/trails of data to be gen
                                 %default:10,000

params.saveit = true;        %saves relevent data if true
                                 %default:false

params.plotit = true;        %plots accuracy vs time
                                 %default:true

%params.pass=true;           %passes over for loop, if plotit=true, then plots
                             %first before terminating script. (if
                             %scores.mat exists, it will load and plot it
                                 %default:false
                             
%params.append=false;        %appends generated data to existing files,
                             %specified by maxsamples and trials
                                 %default:false
  
%params.gen=true;            %generates data specified by maxsamples and trials
                                 %default:false

params.smoothin=true;       %smoothins time series data on train and test set
                                 %default:false
 
                                 
if(isempty(tLdata) && params.gen==false)
    load('tLdata.txt');
    load('tRdata.txt');
elseif(~exist('tLdata.txt','file') || ~exist('tLdata','var'))
    tLdata=[];
    tRdata=[];
    params.gen=true;
    params.saveit = true;
end

[score_models]=testModels(tLdata,tRdata,params);


%nWins = max(size(score_models))-1;
%plots(score_models,100,nWins);


