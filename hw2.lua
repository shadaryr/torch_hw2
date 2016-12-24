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
--  Define our neural network
--  ****************************************************************

local model = nn.Sequential()
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
model:add(nn.Linear(32*4*4, 64))             -- fully connected layer (matrix multiplication between input and weights). gets a 32*4*4 vector, outputs 64 neurons. parameters (32*4*4+1)*64 (the +1 is bias)
model:add(cudnn.ReLU(true))
model:add(nn.Dropout(0.5))                      --Dropout layer with p=0.5
model:add(nn.Linear(64, #classes))            -- 10 is the number of outputs of the network (in this case, 10 digits) (64+1)*10
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
        local x = data:narrow(1, i, batchSize):cuda()
        local yt = labels:narrow(1, i, batchSize):cuda()
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

epochs = 25
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