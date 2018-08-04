function [tL,tR,lens] = GenData(maxSamples,trails, append, beta)
%{
GenData will generate drift diffusion model EEG data n times and stack
them into one matrice structured like
tL    l               tR    l       
[..L..1]   1           [..R..0]   1   
[..L..0]   2           [..R..1]   2   
[..L..1]   3           [..R..0]   3   

...        .             ...      .    
[..L..0]   trails      [..R..1]   trails  

padded with nans. where L,R are the EEG signal data and n is the number of trails(samples). GenData then saves the
data into the same folder as this file called tRdata.txt and tLdata.txt

maxSamples(int):  max number of time samples to be aloud (columns)

trials(int):      number of trials to be generated (rows)

append(bool):     true: appends generated data to the data saved by a
                        previous call of GenData()
                  false: creates fresh ascii txt files for both tR and tL

tL (matrix):      total Right EEG data, with last column being a label
                  (trails x maxSamples) 
tR (matrix):      total Left EEG data, with last colummn being a label
                  (trails x maxSamples)

labeled data: 1 : threshold beta reached
              0 : threshold beta not reached

Labels will automatically be set to be balanced, equal left/right successes
inputs for making the data are hardcoded 
%}
if (~exist('maxSamples','var')), maxSamples=5000; end
if (~exist('trails','var')), trails=10000; end
if (~exist('append','var')), append=false; end
if (~exist('beta','var')), beta = .15; end

params.c=.05;

tL = NaN(trails,maxSamples);  
tR = NaN(trails,maxSamples);
labels = zeros(trails,2);
len = 0;
lens = zeros(1,trails);

for i=1:trails
    [~,l,r]=myModel([.04,.05], [.5,.6],[beta beta],[],params);
    %{
    function [allX, allX_s, allX_v, params] = ...
     RunModel (I, k, beta, nIters, params)
    %}
    [L,R] = labelIt(l,r,beta);
    label = [L,R];
    fprintf('working on trail: %d \n',i);
    
    if(mod(i,2) == 0)
        while (label(1,1) == 0)  
            [~,l,r]=myModel([.04,.05], [.5,.6],[beta beta],[],params);
            [L,R] = labelIt(l,r,beta);
            label = [L,R];
        end
    else
        while (label(1,1) == 1)  
            [~,l,r]=myModel([.04,.05], [.5,.6],[beta beta],[],params);
            [L,R] = labelIt(l,r,beta);
            label = [L,R];
        end
    end
    
    len = max(size(l));
    if (len >= maxSamples)
        temp = (len-maxSamples)+1;
        tL(i,:)=l(1,temp:end);
        tR(i,:)=r(1,temp:end);
    else
        temp = (maxSamples-len)+1;
        tL(i,temp:end)=l(1,:);
        tR(i,temp:end)=r(1,:);
    end
    labels(i,:)=label;
    
    lens(1,i) = len;
end
%fprintf('Mean trail length: %f \n', mean_length);
%fprintf('Standard deviation of the trail length: %f \n', std_length);

if(exist('lenstat.mat','file'))
    temp = lens;
    load lenstat.mat lens
    lens = [temp lens];
end

std_length = std(lens);
mean_length = mean(lens);
save('lenstat.mat','lens','std_length','mean_length');

tL = [tL labels(:,1)];
tR = [tR labels(:,2)];

if ((append == true) && exists('tLdata.txt','file'))
    disp('Data has been appended to files tLdata.txt and tRdata.txt');
    save('tLdata.txt','tL','-ascii','-append');
    save('tRdata.txt','tR','-ascii','-append');
else
    disp('Data created and saved...');
    save('tLdata.txt','tL','-ascii');
    save('tRdata.txt','tR','-ascii');
end

if ((append == true) && ~exists('tLdata.txt','file'))
    disp('Sorry, the file doesnt exist and wasnt appended.');
    disp('Files tLdata.txt and tRdata.txt have been created and saved.');
    disp('Now that they exist, you can append with the next iteration of GenData');
end

end

function [L, R] = labelIt (l,r,beta)
%if one reaches beta, then labeled as 1, else 0
    Ltemp = isempty(find(l>=beta));
    Rtemp = isempty(find(r>=beta));
    if (Ltemp == 1)
        L = 0;
    else
        L = 1;
    end
    
    if (Rtemp == 1)
        R = 0;
    else
        R = 1;
    end
end