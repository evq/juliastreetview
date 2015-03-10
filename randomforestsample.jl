using Images
using DataFrames
using DecisionTree

#typeData could be either "train" or "test.
#labelsInfo should contain the IDs of each image to be read
#The images in the trainResized and testResized data files
#are 20x20 pixels, so imageSize is set to 400.
#path should be set to the location of the data files.

function read_data(typeData, labelsInfo, imageSize, path)
 #Intialize x matrix
 x = zeros(size(labelsInfo, 1), imageSize)

 for (index, idImage) in enumerate(labelsInfo[:ID]) 
  #Read image file 
  nameFile = "$(path)/$(typeData)Resized/$(idImage).Bmp"
  img = imread(nameFile)

  #Convert img to float values 
  #temp = float32sc(img)

  #Convert color images to gray images
  #by taking the average of the color scales. 
  #if ndims(temp) == 3
  # temp = mean(temp.data, 1)
  #end

  temp = convert(Image{Gray}, img)
    
  #Transform image matrix to a vector and store 
  #it in data matrix 
  x[index, :] = reshape(temp, 1, imageSize)
 end 
 return x
end

imageSize = 400 # 20 x 20 pixel

#Set location of data files, folders
path = "./data"

#Read information about training data , IDs.
labelsInfoTrain = readtable("$(path)/trainLabels.csv")

#Read training matrix
xTrain = read_data("train", labelsInfoTrain, imageSize, path)

#Read information about test data ( IDs ).
labelsInfoTest = readtable("$(path)/sampleSubmission.csv")

#Read test matrix
xTest = read_data("test", labelsInfoTest, imageSize, path)

#Get only first character of string (convert from string to character).
#Apply the function to each element of the column "Class"
yTrain = map(x -> x[1], labelsInfoTrain[:Class])

#Convert from character to integer
yTrain = int(yTrain)

model = build_forest(yTrain, xTrain, 20, 50, 1.0)

predTest = apply_forest(model, xTest)

#Convert integer predictions to character
labelsInfoTest[:Class] = char(predTest)

#Save predictions
writetable("juliaSubmission.csv", labelsInfoTest, separator=',', header=true)

accuracy = nfoldCV_forest(yTrain, xTrain, 20, 50, 5, 1.0);
println ("5 fold accuracy: $(mean(accuracy))")
