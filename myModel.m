function [allX, allX_s, allX_v, params] = ...
    myModel (I, k, beta, nIters, params)
%{
 function [allX, allX_s, allX_v, params] = ...
     RunModel (I, k, beta, nIters, params)
 
 The function runs a revised version of the model described in Schurger et
 al. PNAS 2012 (adapted from Usher & McClelland Psychol Rev 2001). 
 The Schurger model is
   dx = (I-k*x)*deltaT + c*xi(j)*sqrtDeltaT;
 We use
   dx_s = (I(1)-k(1)*x_s)*deltaT + c(1)*xi_s(j)*sqrtDeltaT;
   dx_v = (I(2)-k(2)*x_s)*deltaT + c(2)*xi_v(j)*sqrtDeltaT;
 Where x_s and x_v both start at 0. We stop when the maximum between x_s
 and x_v reaches beta (or nIters is reached).
 So the function runs the process for generating the Schurger model, the
 one for generating the value-based model and then weight-sums them into a
 resulting model. It is assumed that the weight-summation includes
 negligible noise.
    
 'I'                is the drift rate
 'k'                is the leak (exponential decay)
 'w'                is the wieghts in the summation of the models into the resulting mode.
 'beta'             is the thresholds
 'nIters'           is the number of iteraitons to run the model.
    
 All of these entries are either scalar, if they are to be the same
 for both models, or vectors, if they differ between the models.
 The function expects either 'beta' or 'nIters', but not both. In fact, if
 'beta' is input, it does not read 'nIters'. If beta is given, the
 function returns the number of iterations until the threshold is reached
 or inf, if more than 'maxIters'. If nIters is given (i.e., 'beta' is []),
 it is the number of iterations to run the model, regardless of threshold. 
 'params' includes additional, secondary, parameters used by the model,
 which could be updated too.
 The function returns 'allX', 'allX_s', and 'allX_v'. It also returns the
 'params' struct in its ultimate form.
%}
    
if (~exist('params','var')), params=[]; end
[c, deltaT, maxIters, params] = GetParams (params);
sqrtDeltaT = sqrt(deltaT);
if (isempty(beta))
    % Run model 'nIters' times
    beta=inf;
else
    % Run model until threshold (or maxIters) is reached
    nIters=maxIters;
end
nModels=2;
if (length(I)==1), I=repmat(I,1,nModels); end
if (length(k)==1), k=repmat(k,1,nModels); end
if (length(c)==1), c=repmat(c,1,nModels); end
xi_c_s=randn(1,nIters)*c(1);
xi_c_v=randn(1,nIters)*c(2);
allX=NaN(1,nIters);
allX_s=allX;
allX_v=allX;
x_s=0; x_v=0;
for j=1:nIters
    % 'x_s' (for Schurger) gets updated without value. This could be viewed
    % as the integration process happenning in the SMA and picked up when
    % measuring the RP in Cz
    dx_s = (I(1)-k(1)*x_s)*deltaT + xi_c_s(j)*sqrtDeltaT;
    x_s=x_s+dx_s;
    allX_s(j)=x_s;
    % 'x_v' (for value-based) gets updated with v*e. This could be viewed
    % as the integration process happenning in a value-related region such
    % as the VMPFC
    dx_v = (I(2)-k(2)*x_v)*deltaT + xi_c_v(j)*sqrtDeltaT;
    x_v=x_v+dx_v ;
    allX_v(j)=x_v;
    % 'x' is sum of x_s and x_v
    x = max(x_s,x_v);
    allX(j)=x;
    if (x>beta)
        allX_s=allX_s(1:j);
        allX_v=allX_v(1:j);
        allX=allX(1:j);
        break;
    end
end

end

% ======================== Internal subfunctions ==========================
function [c, deltaT, maxIters, params] = GetParams (params)
% If the parameters are fields of 'params', extract them. Otherwise, give
% them their default values.

% Noise scaling factor (STD of 0-mean noise)
if (isfield(params,'c'))
    c=params.c;
else
    c = 0.05; 
    params.c=c;
end
% Time step used in simulation
if (isfield(params,'deltaT'))
    deltaT=params.deltaT; 
else
    deltaT = 0.001;
    params.deltaT=deltaT;
end
% Maximum number of iterations allowed in model
if (isfield(params,'maxIters'))
    maxIters=params.maxIters; 
else
    maxIters=100000;
    params.maxIters=maxIters;
end
end
    
%{
>> params.c=.05; [~,l,r]=RunModel([.04,.05], [.5,.6],[.15 .15],[],params);
>> t=(0:length(l)-1)./1000;
>> figure; subplot(121); plot(t,l); subplot(122); plot(t,r);
https://www.mathworks.com/help/stats/
https://www.mathworks.com/help/nnet/ug/deep-learning-in-matlab.html
https://www.mathworks.com/help/nnet/examples/time-series-forecasting-using-deep-learning.html
%}





