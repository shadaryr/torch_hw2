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
        if 0 == permutation[i] % 3  then hflip(input[i]) end 
		if 1 == permutation[i] % 3  then randomcrop(input[i], 10, 'reflection') end
		if 2 == permutation[i] % 3  then randomcrop(input[i], 10, 'zero') end
      end -- and if 2== %3 -> do nothing.
    end -- in the expectancy - will train on a 3 times bigger set than the original training set, without really saving all augmented data! just by doing a lot of epochs - beacuse we do this in each epoch for each image all over again
    self.output:set(input:cuda())
    return self.output
  end
end
 


--[[
-- Test hflip
local im = image.load('img.jpg') --im will hold a tensor
image.save('imghflip.jpg', image.hflip(im))

-- Test reflection
local im = image.load('img.jpg')
local I = randomcrop(im, 10, 'reflection')
print(I:size(),im:size())
image.save('img2ReflectionCrop.jpg', I)

-- Test zero padding with crop
local im = image.load('img.jpg')
local I = randomcrop(im, 10, 'zero')
print(I:size(),im:size())
image.save('img2ZeroCrop.jpg', I)
]]

--  ****************************************************************
--  Define our neural network
--  ****************************************************************
-- all the calculation near the layers are the output size of the layer

--model:add(nn.BatchFlip():float()) --data augmentation layer

local model = nn.Sequential()
model:add(nn.BatchFlip():float())
model:add(cudnn.SpatialConvolution(3, 32, 5, 5)) -- 3 input image channel, 32 output channels, 5x5 convolution kernel. owidth=floor((32+2*0-5)/1 +1)=28. same goes to ohight. output- 28*28*32
model:add(cudnn.SpatialMaxPooling(2,2,2,2))      -- A max-pooling operation that looks at 2x2 windows and finds the max. floor(28+2*0-2)/2+1)*floor(28+2*0-2)/2+1)*32(depth do not change) = 14*14*32
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
model:add(nn.Dropout(0.2))                      --Dropout layer with p=0.5
--model:add(nn.Linear(64, #classes))            -- 10 is the number of outputs of the network (in this case, 10 digits) (64+1)*10
model:add(nn.Linear(512, #classes))            -- 10 is the number of outputs of the network (in this case, 10 digits) (512+1)*10
model:add(nn.LogSoftMax())                     -- converts the output to a log-probability. Useful for classificati

model:cuda()
criterion = nn.ClassNLLCriterion():cuda()


w, dE_dw = model:getParameters()
print('Number of parameters:', w:nElement())
print(model)

function shuffle(data,ydata) --shuffle data function
    local RandOrder = torch.randperm(data:size(1)):long()
    return data:index(1,RandOrder), ydata:index(1,RandOrder)
end

--  ****************************************************************
--  Training the network
--  ****************************************************************
require 'optim'

local batchSize = 128
local optimState = {}

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
        --local x = data:narrow(1, i, batchSize):cuda()
        --local yt = labels:narrow(1, i, batchSize):cuda()
        local x = data:narrow(1, i, batchSize)
        local yt = labels:narrow(1, i, batchSize)
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
        
            optim.adam(feval, w, optimState)
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

epochs = 90
trainLoss = torch.Tensor(epochs)
testLoss = torch.Tensor(epochs)
trainError = torch.Tensor(epochs)
testError = torch.Tensor(epochs)

--reset net weights
model:apply(function(l) l:reset() end)

timer = torch.Timer()

for e = 1, epochs do
    trainData, trainLabels = shuffle(trainData, trainLabels) --shuffle training data
    trainLoss[e], trainError[e] = forwardNet(trainData, trainLabels, true)
    testLoss[e], testError[e], confusion = forwardNet(testData, testLabels, false)
    
    if e % 5 == 0 then
        print('Epoch ' .. e .. ':')
        print('Training error: ' .. trainError[e], 'Training Loss: ' .. trainLoss[e])
        print('Test error: ' .. testError[e], 'Test Loss: ' .. testLoss[e])
        print(confusion)
    end
end

--  ****************************************************************
--  printing plots
--  ****************************************************************

plotError(trainError, testError, 'Classification Error')
plotLoss (trainLoss, testLoss, 'Classification Loss')