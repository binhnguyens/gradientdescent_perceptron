%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% BME777: LAB 2: Linear Discriminant Functions.
% Acknowledgement: We thankfully acknowledge UCI Machine Learning Repository for the 
% dataset used in this lab exercise.
% Indian Liver Patient Dataset.
% Link: https://archive.ics.uci.edu/ml/datasets/ILPD+%28Indian+Liver+Patient+Dataset%29#

% Class1: Liver patient. Class2: non Liver patient.
% DataLab2_1: Features: TP Total Proteins and ALB Albumin with modification for problem simplification. 
% Features 8-9. 
% DataLab2_2: Features: TP Total Proteins and A/G Ratio	Albumin and
% Globulin Ratio. Features 8-10.
% 50 samples were extracted for each class.

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Inputs:
% 1. Data: 100x3 dataset. The first column contains the feature x1, the second
% column contains the feature x2. The class labels are given in the third
% column.
% 2. ClassSplit: Threshold where classes are divided. See the third
% column of the Data to choose the correct threshold.
% 3. DataSplitRate: Threhold to split the data in each class into training and testing data.
% For e.g., DataSplitRate = 0.4 ==> 40% data of class 1,2 is for training.
% 60% of the data is for testing.
% 4. InitialParameterSet: Initial values of the set of parameters. For
% e.g., InitialParameterSet = [0 0 1].
% 5. LearningRate: Learning rate when updating the algorithm.
% 5. Theta: The expected cost that the optimized parameter set may give.
% 6. MaxNoOfIteration: the maximum number of iterations the algorithm may run.
%
% Output:
% 1: TrainedParameterSet: The set of optimized parameters.
% 2: NoOfIteration: The number of iteration when the algorithm converges.
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Example of use:
% load DataLab2_1.mat
% Data = DataLab2_1;
% ClassSplit = 50;
% DataSplitRate = 0.4;
% InitialParameterSet = [0 0 1];
% LearningRate = 0.01;
% Theta = 0;
% MaxNoOfIteration = 300;
% [OptimizedParameterSet,NoOfIteration] = ...
% lab2(Data,ClassSplit,DataSplitRate, ... 
% InitialParameterSet,LearningRate,Theta,MaxNoOfIteration);
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

close all;
clear;


load DataLab2_2.mat;
Data = DataLab2_2;
ClassSplit = 50;
DataSplitRate = 0.4;
InitialParameterSet = [0 0 1];
LearningRate = 0.01;
Theta = 0;
MaxNoOfIteration = 1000;
% [OptimizedParameterSet,NoOfIteration] = lab2(Data,ClassSplit,DataSplitRate, InitialParameterSet,LearningRate,Theta,MaxNoOfIteration);


% function [TrainedParameterSet,NoOfIteration] =lab2(Data,ClassSplit,DataSplitRate,InitialParameterSet,LearningRate,Theta,MaxNoOfIteration)



[Len,~] = size(Data);

% Split the data into two classes based on ClassSplit. 
Class1 =Data (1:ClassSplit,:);
Class2 =Data (ClassSplit+1:end,:);

% Calculate the number of training samples.
Train_Num1 =(ClassSplit)*DataSplitRate; %training sample of class 1
Train_Num2 =(ClassSplit)*(DataSplitRate); %training sample of class 2

% Split the data in class 1 into training and testing sets. 

for i =1:(Train_Num1)
    Train_Class1 (i,:)= Class1(i,:);
end

z=1;
for i =(Train_Num1+1):numel(Class1(:,1))
    Test_Class1 (z,:)= Class1(i,:);
    z=z+1;
end



% Split the data in class 2 into training and testing sets.

for i =1:(Train_Num2)
    Train_Class2 (i,:)= Class2(i,:);
    
    %This is done so that the label values become 1
    if Train_Class2(i,3) == 2
        Train_Class2 (i,3) = 1;
    end
    
end

z=1;
for i =(Train_Num2+1):numel(Class2(:,1))
    Test_Class2 (z,:)= Class2(i,:);
    
%     %This is done so that the label values become 1
%     if Test_Class2(z,3) == 2
%         Test_Class2 (z,3) = 1;
%     end
    
    z=z+1;
end


% Test_Class2=-Test_Class2;
Train_Class2=-Train_Class2;

% Prepare the training data including all training samples of classs 1 and
% 2.
Train_Data = zeros(Train_Num1 + Train_Num2,3);
Train_Data(1:Train_Num1,1) = 1; 
Train_Data(Train_Num1+1:Train_Num1+Train_Num2,1) = -1;
Train_Data(1:Train_Num1,2:3) = Train_Class1(:,1:2);
Train_Data(Train_Num1+1:Train_Num1 + Train_Num2,2:3)= Train_Class2(:,1:2);



% Prepare the test data including all test samples of class 1 and
% 2.
if((Train_Num1+Train_Num2)~=(length(Class1)+length(Class2)))
	Test_Data(1:length(Test_Class1) + length(Test_Class2),1) = 1;
	Temp = [Test_Class1(:,1:2); Test_Class2(:,1:2)];
	Test_Data(:,2:3) =Temp;
end



% Implement basic gradient algorithm.
OptParams = InitialParameterSet; %a=initialParameterSet=[0 0 1]
PerceptronFunction = zeros(MaxNoOfIteration,1); %MaxNoOfIteration=300
Criterion = 1;
NoOfIteration = 1;

while ((Criterion>Theta)) %interface theta=0, so while(1>0)
    GradientOfCost = zeros(1,3); %placeholder
    
    % Update the PerceptronFunction and The GradientOfCost.
    for i=1:(Train_Num1 + Train_Num2)    
        % Use the current OptParams and the ith train data to predict the class.
        PredictedValue = (OptParams) *  transpose(Train_Data(i,:)); 
        
        if (PredictedValue <= 0)
            GradientOfCost = GradientOfCost - Train_Data(i,:);
            %Summing of the misclassified 
            %gradient Jp = sum (-y)
            
            PerceptronFunction(NoOfIteration) = -PredictedValue+ PerceptronFunction(NoOfIteration);

            %Represents the jp(a) = sum (-a^t * y);
            %y represents GradientOfCost or Train_Data(i,:)
            
        end
    end
    
    % Update the optimized parameters.
    OptParams = OptParams - LearningRate*GradientOfCost; %a(k+1)=a(k)+miu(k)*gradient
    
    % Update the value of the criterion to stop the algorithm.
    % |n(k)*gradient| < threshold
   
    
    Criterion=norm(GradientOfCost*LearningRate);
    
    
%     norm(GradientOfCost*LearningRate)
   
    
    % Break the algorithm when the NoOfIteration = MaxNoOfIteration.
    if(NoOfIteration == MaxNoOfIteration)
        break;
    end
    NoOfIteration = NoOfIteration + 1;

end


% Plot data of class 1, class 2 and the estimated boundary.

% Plot the values of the perceptron function.
%%%%
%%%%
%%%%
%%%%
figure;
plot(PerceptronFunction);
title('Perceptron Function'); 
axis('tight');
ylabel('Perception Function Value');xlabel('Iteration');

% Plot the values Feature space
%%%%
%%%%
%%%%
%%%%
figure;
scatter (Class1(:,1),Class1(:,2)); %class 1 
hold on;
scatter (Class2(:,1),Class2(:,2)); %class 2
hold on;

% %%%
% figure;
% scatter (Test_Class1(:,1),Test_Class1(:,2)); %class 1 
% hold on;
% scatter (Test_Class2(:,1),Test_Class2(:,2)); %class 2
% hold on;

% w0 w1 w2
% x0+x1+x2
% x2 = -(x0+x1)
z = @(x) (OptParams(1) + OptParams(2) * x)/ -OptParams(3) ;
values = -5:10;
plot (values,z(values))
title('Training Class Data')
legend('Train Class 1', 'Train Class 2', 'Boundary');
ylabel('x2');xlabel('x1');



% Calculate the classification accuracy of the predictions on the test data.

%NoOfAccuracy is used to count the number of -ve 
NoOfAccuracy = 0;

%a*y using the optimzed opt params
test = OptParams* transpose(Test_Data);

%Holder is used to count the number of +ve classifications
holder=0;

if((Train_Num1+Train_Num2)~=(length(Class1)+length(Class2)))
	for j=1:(length(Test_Data)/2)
        
        %used to check and count the number of times -ve and +ve values
        %come from the jp = a*y (perceptron function)
        if test(j) >=0
            NoOfAccuracy = NoOfAccuracy+1;
        end
        
    end
    for j=(length(Test_Data)/2)+1:length(Test_Data);
        
        %used to check and count the number of times -ve and +ve values
        %come from the jp = a*y (perceptron function)
        if test(j) <=0
            NoOfAccuracy = NoOfAccuracy+1;
        end
        
    end
    
    
end
	
%Very unorthodox
%abs((holder - NoOfAccuracy)/2)/length(Test_Data) 
%this represents the abs(+ve count - the -ve count)/2
%this is done to find the number of WRONG classifications
%e.g you know the perceptron is suppose to have 50% +ve and 50% -ve, yet
%you have 36 right and 24 wrong. that means you have an error of 6


ClassificationAccuracy = NoOfAccuracy/length(Test_Data);

% NoOfIteration

TrainedParameterSet = OptParams

% end



