function returnAvgError()
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
	testData = testData:cuda()
	testLabels = testLabels:cuda()

	--load the model (the trained net)
	model = torch.load('HW2_network_v2.t7')
	model:evaluate() --turn off drop out

	--calculating the estimated labels with the trained nn
	local y_hat = model:forward(testData)
	-- creating and calculating confusion matrix
	local confusion = optim.ConfusionMatrix(torch.range(0,9):totable())
	confusion:batchAdd(y_hat,testLabels)

	-- calculating average error
	confusion:updateValids()
	local avgError = 1 - confusion.totalValid

	-- returning the average error
	return avgError
end

print ('avgError is :',returnAvgError())