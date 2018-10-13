# DeepLearningWithJulia
# this is the second lecture code; Using Flux, a neural network is described to recognize MNIST database of labeled digits.

using Flux, Flux.Data.MNIST, Statistics
using Flux: onehotbatch, onecold, crossentropy, throttle
using Base.Iterators: repeated

imgs=MNIST.images() #array of arrays (each array is an image.)
size(imgs)
typeof(imgs[1])
X=hcat(float.(reshape.(imgs,:))...) # we use X since the model won't understand the images. 
labels=MNIST.labels()
Y=onehotbatch(labels, 0:9) # used to encode the values into true/false. this is changing the classes into probabilities so I can use crossentropy
#now we use chain as a function from Flux. we tell it how many layers we have, actually it defines .  Dense type means it is full connected layer (all nodes are connected.) it initializes the weights randomly.
m= Chain( #define the model
    Dense(28^2, 32, relu), # weights from 784 to 32 nodes and use relu activation.
    Dense(32, 10), #take from 32 to 10 and don't do activation since we'll do a softmax.
    softmax) # we'll do a softmax to the 10 output. softmax actually normalizes the output activation to a probability 0 to 1.
# use m(X[:,15]) to feedforward.
loss(x,y)=crossentropy(m(x), y) # this is a function definition.
#now we have the loss function define. we can use it as in
#loss(X[:,15],Y[:,15]) this will output a tracked variable so it initializes only once and then updates .

accuracy(x,y)= mean(onecold(m(x)) .==onecold(y))

loss(X,Y)
Flux.train!(loss,dataset, opt)
loss (X,Y)

dataset= repeated((X,Y),10)) # now we have repeated 10 times of x,y. and we used dataset in the model
#which means we feed all 60000 images to the model for each iteration and we repeat that 10 times.
opt = ADAM (params(m)) #using ADAM optimizer.
# params looks at all model parameters so we change these parameters to optimize the loss function.
Flux.train!(loss, dataset, opt, cb=throttle(evalcb,10)) 
    #loss is the function to optimize,
    #dataset contains all images.
    #opt is the algorithm we use to optimize.
