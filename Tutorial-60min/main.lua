require 'paths'

if (not paths.filep("cifar10torchsmall.zip")) then
    os.execute('wget -c https://s3.amazonaws.com/torch7/data/cifar10torchsmall.zip')
    os.execute('unzip cifar10torchsmall.zip')
end

-- unseriable object / unpack object from file  
trainset = torch.load('cifar10-train.t7')
testset = torch.load('cifar10-test.t7')

-- define class 
classes = {'airplane', 'automobile', 'bird', 'cat',
           'deer', 'dog', 'frog', 'horse', 'ship', 'truck'}

print(trainset)
print(#trainset.data)

print(#trainset.data[100]) -- display the 100-th image in dataset
print(classes[trainset.label[100]])

-- 
-- pre process object trainset
-- ignore setmetatable for now, it is a feature beyond the scope of this tutorial. It sets the index operator.
setmetatable(trainset, 
    {__index = function(t, i) 
                    return {t.data[i], t.label[i]} 
                end}
);
trainset.data = trainset.data:double() -- convert the data from a ByteTensor to a DoubleTensor.

function trainset:size() 
    return self.data:size(1) 
end

-- test object trainset 
print(trainset:size()) -- just to test
print(trainset[33]) -- load sample number 33.
print(#trainset[33][1])

redChannel = trainset.data[{ {}, {1}, {}, {}  }] -- this picks {all images, 1st channel, all vertical pixels, all horizontal pixels}
print(#redChannel)