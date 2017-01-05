function returnAvgError()
	require 'optim'
	require 'torch'
	require 'image'
	require 'nn'
	require 'cunn'
	require 'cudnn'	

	--loading the data, converting to float tensors, dividing into test and train tensors.
	local trainset = torch.load('cifar.torch/cifar10-train.t7')
	local testset = torch.load('cifar.torch/cifar10-test.t7')

	local classes = {'airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck'}

	local trainData = trainset.data:float() -- convert the data from a ByteTensor to a float Tensor.
	local trainLabels = trainset.label:float():add(1)
	local testData = testset.data:float()
	local testLabels = testset.label:float():add(1)

	--normalizing our test data w.r.t. the train mean and std
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


	-- tranfering to cuda
	--testData = testData:cuda()
	--testLabels = testLabels:cuda()
	
	------DATA AUGMENTATION -- BATCHFLIP-----------
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

	local confusion = optim.ConfusionMatrix(classes)
	local lossAcc = 0
	local numBatches = 0
	local batchSize = 32
	criterion = nn.CrossEntropyCriterion():cuda()
	--load the model (the trained net)
	model = torch.load('HW2_network_v3.t7')
	model:evaluate() --turn off drop out

	--calculating the estimated labels with the trained nn
	for i = 1, testData:size(1) - batchSize, batchSize do
        numBatches = numBatches + 1
        local x = testData:narrow(1, i, batchSize)
        local yt = testLabels:narrow(1, i, batchSize)
        local y = model:forward(x)
        local err = criterion:forward(y, yt)
        lossAcc = lossAcc + err
        confusion:batchAdd(y,yt)
    end
	
	-- calculating average error
	confusion:updateValids()
	local avgError = 1 - confusion.totalValid

	-- returning the average error
	return avgError
end

print ('avgError is :',returnAvgError())