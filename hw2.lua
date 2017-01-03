require 'torch'
require 'image'
require 'nn'
require 'cunn'
require 'cudnn'

function saveTensorAsGrid(tensor,fileName)
	local padding = 1
	local grid = image.toDisplayTensor(tensor/255.0, padding)
	image.save(fileName,grid)
end


--  ****************************************************************
--  Loading the data from cifar10
--  ****************************************************************

local trainset = torch.load('cifar.torch/cifar10-train.t7')
local testset = torch.load('cifar.torch/cifar10-test.t7')

local classes = {'airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck'}

local trainData = trainset.data:float() -- convert the data from a ByteTensor to a float Tensor.
local trainLabels = trainset.label:float():add(1)
local testData = testset.data:float()
local testLabels = testset.label:float():add(1)


--  ****************************************************************
--  Normalizing the data
--  ****************************************************************

-- Load and normalize data:

local mean = {}  -- store the mean, to normalize the test set in the future
local stdv  = {} -- store the standard-deviation for the future
for i=1,3 do -- over each image channel
    mean[i] = trainData[{ {}, {i}, {}, {}  }]:mean() -- mean estimation
    trainData[{ {}, {i}, {}, {}  }]:add(-mean[i]) -- mean subtraction
    
    stdv[i] = trainData[{ {}, {i}, {}, {}  }]:std() -- std estimation
    trainData[{ {}, {i}, {}, {}  }]:div(stdv[i]) -- std scaling
end

-- Normalize test set using same values

for i=1,3 do -- over each image channel
    testData[{ {}, {i}, {}, {}  }]:add(-mean[i]) -- mean subtraction    
    testData[{ {}, {i}, {}, {}  }]:div(stdv[i]) -- std scaling
end

--  ****************************************************************
--  data augmentation
--  ****************************************************************

local function hflip(x)
--[[
Flips image src horizontally (left<->right). 
If dst is provided, it is used to store the output image. 
Otherwise, returns a new res Tenso
]]
   return torch.random(0,1) == 1 and x or image.hflip(x)
end

local function vflip(x)
--[[
Flips image src horizontally (left<->right). 
If dst is provided, it is used to store the output image. 
Otherwise, returns a new res Tenso
]]
   return torch.random(0,1) == 1 and x or image.vflip(x)
end

local function randomcrop(im , pad, randomcrop_type)
   if randomcrop_type == 'reflection' then
      -- Each feature map of a given input is padded with the replication of the input boundary
      module = nn.SpatialReflectionPadding(pad,pad,pad,pad):float() 
   elseif randomcrop_type == 'zero' then
      -- Each feature map of a given input is padded with specified number of zeros.
	  -- If padding values are negative, then input is cropped.
      module = nn.SpatialZeroPadding(pad,pad,pad,pad):float()
   end
	
   local padded = module:forward(im:float())
   local x = torch.random(1,pad*2 + 1)
   local y = torch.random(1,pad*2 + 1)
   --image.save('img2ZeroPadded.jpg', padded)

   --return torch.random(0,1) == 1 and x or padded:narrow(3,x,im:size(3)):narrow(2,y,im:size(2))
   return padded:narrow(3,x,im:size(3)):narrow(2,y,im:size(2))
end

do -- data augmentation module
  local BatchFlip,parent = torch.class('nn.BatchFlip', 'nn.Module')

  function BatchFlip:__init()
    parent.__init(self)
    self.train = true --so the data augmentation will only happen in the training phase!!!
  end

  function BatchFlip:updateOutput(input)
    if self.train then
      local permutation = torch.randperm(input:size(1))
      for i=1,input:size(1) do
		local mod = permutation[i] % 4
        if 0 == mod  then image.hflip(input[i]) end 
		if 1 == mod  then randomcrop(input[i], 10, 'reflection') end
		if 2 == mod  then randomcrop(input[i], 10, 'zero') end
      end -- and if mod ==3 -> do nothing.
    end
    self.output:set(input:cuda())
    return self.output
  end
end
 

--  ****************************************************************
--  Define our neural network
--  ****************************************************************
-- all the calculation near the layers are the output size of the layer
local model = nn.Sequential()
--model:add(nn.BatchFlip():float())--data augmentation layer
model:add(cudnn.SpatialConvolution(3, 64, 5, 5, 1, 1, 2, 2))
model:add(nn.SpatialBatchNormalization(64))    --Batch normalization will provide quicker convergence
model:add(cudnn.ReLU(true))
model:add(cudnn.SpatialConvolution(64, 32, 1, 1)) -- 
model:add(nn.SpatialBatchNormalization(32))    --Batch normalization will provide quicker convergence
model:add(cudnn.ReLU(true))
model:add(cudnn.SpatialConvolution(32, 32, 1, 1)) -- 
model:add(nn.SpatialBatchNormalization(32))    --Batch normalization will provide quicker convergence
model:add(cudnn.ReLU(true))
model:add(cudnn.SpatialMaxPooling(3,3,2,2):ceil()) --
model:add(nn.Dropout(0.2))
model:add(cudnn.SpatialConvolution(32, 32, 5, 5, 1, 1, 2, 2)) -- 
model:add(nn.SpatialBatchNormalization(32))    --Batch normalization will provide quicker convergence
model:add(cudnn.ReLU(true))
model:add(cudnn.SpatialConvolution(32, 32, 1, 1)) -- 
model:add(nn.SpatialBatchNormalization(32))    --Batch normalization will provide quicker convergence
model:add(cudnn.ReLU(true))
model:add(cudnn.SpatialConvolution(32, 32, 1, 1)) -- 
model:add(nn.SpatialBatchNormalization(32))    --Batch normalization will provide quicker convergence
model:add(cudnn.ReLU(true))
model:add(cudnn.SpatialAveragePooling(3,3,2,2):ceil()) --
model:add(nn.Dropout(0.2))
model:add(cudnn.SpatialConvolution(32, 32, 3, 3, 1, 1, 1, 1)) -- 
model:add(nn.SpatialBatchNormalization(32))    --Batch normalization will provide quicker convergence
model:add(cudnn.ReLU(true))
model:add(cudnn.SpatialConvolution(32, 32, 1, 1)) -- 
model:add(nn.SpatialBatchNormalization(32))    --Batch normalization will provide quicker convergence
model:add(cudnn.ReLU(true))
model:add(cudnn.SpatialConvolution(32, 10, 1, 1)) -- 
model:add(nn.SpatialBatchNormalization(10))    --Batch normalization will provide quicker convergence
model:add(cudnn.ReLU(true))
model:add(cudnn.SpatialAveragePooling(8,8,1,1):ceil()) --
model:add(nn.View(#classes))

--[[ old architecture
local model = nn.Sequential()
--model:add(nn.BatchFlip():float())--data augmentation layer
model:add(cudnn.SpatialConvolution(3, 32, 5, 5)) -- 3 input image channel, 32 output channels, 5x5 convolution kernel. owidth=floor((32+2*0-5)/1 +1)=28. same goes to ohight. output- 28*28*32
--model:add(cudnn.SpatialMaxPooling(2,2,2,2))      -- A max-pooling operation that looks at 2x2 windows and finds the max. floor(28+2*0-2)/2+1)*floor(28+2*0-2)/2+1)*32(depth do not change) = 14*14*32
model:add(cudnn.ReLU(true))                          -- ReLU activation function
model:add(nn.SpatialBatchNormalization(32))    --Batch normalization will provide quicker convergence
model:add(cudnn.SpatialConvolution(32, 64, 3, 3)) -- gets 14*14*32. 64 filters. 3*3 is the surface of each kernel floor((14+2*0-3)/1 +1)*floor((14+2*0-3)/1 +1)*64=12*12*64
model:add(cudnn.SpatialMaxPooling(2,2,2,2)) --floor(12+2*0-2)/2+1)*floor(12+2*0-2)/2+1)*64(depth do not change) = 6*6*64
model:add(cudnn.ReLU(true))
model:add(nn.SpatialBatchNormalization(64))
model:add(cudnn.SpatialConvolution(64, 32, 3, 3)) --gets 6*6*64. 32 filters. 3*3 is the surface of each kernel floor((6+2*0-3)/1 +1)*floor((6+2*0-3)/1 +1)*32=4*4*32
model:add(nn.View(32*4*4):setNumInputDims(3))  -- reshapes from a 3D tensor of 32x4x4 into 1D tensor of 32*4*4
--model:add(nn.Linear(32*4*4, 64))             -- fully connected layer (matrix multiplication between input and weights). gets a 32*4*4 vector, outputs 64 neurons. (32*4*4+1)*64 (the +1 is bias)
model:add(cudnn.ReLU(true))
model:add(nn.Dropout(0.5))                      --Dropout layer with p=0.5
--model:add(nn.Linear(64, #classes))            -- 10 is the number of outputs of the network (in this case, 10 digits) (64+1)*10
model:add(nn.Linear(512, #classes))            -- 10 is the number of outputs of the network (in this case, 10 digits) (512+1)*10
model:add(nn.LogSoftMax())                     -- converts the output to a log-probability. Useful for classificati
]]


model:cuda()
criterion = nn.CrossEntropyCriterion():cuda()


w, dE_dw = model:getParameters()
print('Number of parameters:', w:nElement())
print(model)

local f = assert(io.open('logFile4.log', 'w'), 'Failed to open input file')
 --print('open the file')
   --f:write('The model is: ')
--print('start print to the log')
   --f:write(model)
   f:write('Number of parameters: ')
   f:write('Description of model: sgd, NO batchflip, dropout 0.2, test and train plots number4')
   f:write(w:nElement())
   f:write('\n The criterion is: CrossEntropyCriterion')
   --f:write(criterionName)
   f:write('\n optim function: crossentropy')
   f:write('\nsgd\n')

function shuffle(data,ydata) --shuffle data function
    local RandOrder = torch.randperm(data:size(1)):long()
    return data:index(1,RandOrder), ydata:index(1,RandOrder)
end

--  ****************************************************************
--  Training the network
--  ****************************************************************
require 'optim'

local batchSize = 32
f:write('batchSize: ')
f:write(batchSize)
f:write('\n')
f:close()
local optimState = {
learningRate = 0.05
}

function forwardNet(data,labels, train)
    --another helpful function of optim is ConfusionMatrix
    local confusion = optim.ConfusionMatrix(classes)
    local lossAcc = 0
    local numBatches = 0
    if train then
        --set network into training mode
        model:training()
    else
        model:evaluate() -- turn of drop-out
    end
    for i = 1, data:size(1) - batchSize, batchSize do
        numBatches = numBatches + 1
        local x = data:narrow(1, i, batchSize):cuda()
        local yt = labels:narrow(1, i, batchSize):cuda()
		--local x = data:narrow(1, i, batchSize)
        --local yt = labels:narrow(1, i, batchSize)
		local y = model:forward(x)
        local err = criterion:forward(y, yt)
        lossAcc = lossAcc + err
        confusion:batchAdd(y,yt)
        
        if train then
            function feval()
                model:zeroGradParameters() --zero grads
                local dE_dy = criterion:backward(y,yt)
                model:backward(x, dE_dy) -- backpropagation
            
                return err, dE_dw
            end
        
            optim.sgd(feval, w, optimState)
        end
    end
    
    confusion:updateValids()
    local avgLoss = lossAcc / numBatches
    local avgError = 1 - confusion.totalValid
    
    return avgLoss, avgError, tostring(confusion)
end

function plotError(trainError, testError, title)
	require 'gnuplot'
	local range = torch.range(1, trainError:size(1))
	gnuplot.pngfigure('testVsTrainError.png')
	gnuplot.plot({'trainError',trainError},{'testError',testError})
	gnuplot.xlabel('epochs')
	gnuplot.ylabel('Error')
	gnuplot.plotflush()
end

function plotLoss(trainLoss, testLoss, title)
	local range = torch.range(1, epochs)
	gnuplot.pngfigure('testVsTrainLoss.png')
	gnuplot.plot({'trainLoss',trainLoss},{'testLoss',testLoss})
	gnuplot.xlabel('epochs')
	gnuplot.ylabel('Loss')
	gnuplot.plotflush()
end
---------------------------------------------------------------------

--  ****************************************************************
--  Executing the network training
--  ****************************************************************

epochs = 700
trainLoss = torch.Tensor(epochs)
testLoss = torch.Tensor(epochs)
trainError = torch.Tensor(epochs)
testError = torch.Tensor(epochs)

--reset net weights
model:apply(function(l) l:reset() end)

timer = torch.Timer()
print "starting epochs"

for e = 1, epochs do
    print('start epoch ' .. e .. ':')
		
    trainData, trainLabels = shuffle(trainData, trainLabels) --shuffle training data
    trainLoss[e], trainError[e] = forwardNet(trainData, trainLabels, true)
    testLoss[e], testError[e], confusion = forwardNet(testData, testLabels, false)
    
    if e % 5 == 0 then
        print('Epoch ' .. e .. ':')
        print('Training error: ' .. trainError[e], 'Training Loss: ' .. trainLoss[e])
        print('Test error: ' .. testError[e], 'Test Loss: ' .. testLoss[e])
        print(confusion)
   else
	 
        print('Epoch ' .. e .. ':')
        print('Training error: ' .. trainError[e], 'Training Loss: ' .. trainLoss[e])
        print('Test error: ' .. testError[e], 'Test Loss: ' .. testLoss[e])
		print('\nAccuracy:', 1-testError[e])
   end
   
   if e == 1 then
      bestError = testError[e]
   end

local WritetrainError = trainError[e]
local WritetrainLoss = trainLoss[e] 
local WritetestError = testError[e]
local WritetestLoss = testLoss[e]
local f = assert(io.open('logFile4.log', 'a+'), 'Failed to open input file')
	if e > 1 then
		print('\nbest Error till this epoch: ')
		print(bestError)
		print('test Error: ')
		print(testError[e])
	if (testError[e] < bestError) then
	    bestError = testError[e]
		print('\nbest Error: ')
		print(bestError)
	    print('save the model')
	    torch.save('HW2_network_v3.t7', model)
	        --f = assert(io.open('logFile.log', 'r'), 'Failed to open input file')
	    f:write('Epoch ' .. e .. ': \n')
	    WritetrainError = trainError[e]
	    WritetrainLoss = trainLoss[e] 
	    WritetestError = testError[e]
	    WritetestLoss = testLoss[e]
	    f:write('Training error: ' .. WritetrainError ..  ' Training Loss: ' .. WritetrainLoss .. '\n')
	    f:write('Test error: ' .. WritetestError .. ' Test Loss: ' .. WritetestLoss ..'\n')
	end
    else
		print('save the model')
		torch.save('HW2_network_v3.t7', model)
		f:write('Epoch ' .. e .. ': \n')
		WritetrainError = trainError[e]
		WritetrainLoss = trainLoss[e] 
		WritetestError = testError[e]
		WritetestLoss = testLoss[e]
		f:write('Training error: ' .. WritetrainError ..  ' Training Loss: ' .. WritetrainLoss .. '\n')
		f:write('Test error: ' .. WritetestError .. ' Test Loss: ' .. WritetestLoss ..'\n')
    end	
    f:close()
end

--  ****************************************************************
--  printing plots
--  ****************************************************************

plotError(trainError, testError, 'Classification Error')

require 'gnuplot'
local range = torch.range(1, epochs)
gnuplot.pngfigure('loss4.png')
gnuplot.plot({'trainLoss',trainLoss},{'testLoss',testLoss})
gnuplot.xlabel('epochs')
gnuplot.ylabel('Loss')
gnuplot.plotflush()

gnuplot.pngfigure('error4.png')
gnuplot.plot({'trainError',trainError},{'testError',testError})
gnuplot.xlabel('epochs')
gnuplot.ylabel('Error')
gnuplot.plotflush()