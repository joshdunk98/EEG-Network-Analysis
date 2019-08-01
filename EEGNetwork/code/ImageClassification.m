%For this file, use the ImgClassify.mat file
%Contains the options and layers needed for model training

%Specify the full path for the images that you are using
rootFolder = fullfile("/Users/jdunkley98/spark-2019/data/Gray-Graphs")

%Specify the subfolders if any
categories = {'AD', 'MCI', 'NC'}

%Create an image datastore that contains info on all of the images
imds = imageDatastore(fullfile(rootFolder, categories), ...
    'IncludeSubfolders',true,'LabelSource','foldernames');

%Create three arrays to keep track of predictions and accuracy of the model
accuracy2 = double.empty(48,1);
YPred2 = string.empty(48,1);
YValidation2 = string.empty(48,1);


for i = 1:48
    %Specify list of indices used for training
    if i == 1
        indices = cell2mat({2:48});
    elseif i == 48
        indices = cell2mat({1:47});
    else
        indices = [cell2mat({1:i-1}), cell2mat({i+1:48})];

    %Split data into training and testing sets
    %Based on leave-one-out cross validation where
    %one subject is used for testing while the other subjects
    %are used for training
    train = subset(imds, indices);
    test = subset(imds, i);
    
    %train the new network
    newNet = trainNetwork(train, layers.Layers, options);

    %Classify the testing image
    YPred2(i) = classify(newNet,test);
    YValidation2(i) = test.Labels;

    %See if the image was categorized correctly
    accuracy2(i) = sum(YPred2(i) == YValidation2(i))/numel(YValidation2(i));
end
