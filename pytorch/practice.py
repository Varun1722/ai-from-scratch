import marimo

__generated_with = "0.23.15"
app = marimo.App(width="medium")


@app.cell
def _():
    import torch

    return (torch,)


@app.cell
def _():
    import marimo as mo

    return (mo,)


@app.cell
def _(torch):
    torch.__version__
    return


@app.cell
def _(torch):
    torch.cuda.is_available()
    return


@app.cell
def _(torch):
    torch.backends.mps.is_available()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ##PyTorch Tensors
    PyTorch tensors are data containers for array-like structures. A scalar is a 0-dimensional tensor (for instance, just a number), a vector is a 1-dimensional tensor, and a matrix is a 2-dimensional tensor. There is no specific term for higher-dimensional tensors, so we typically refer to a 3-dimensional tensor as just a 3D tensor, and so forth.
    """)
    return


@app.cell
def _(torch):
    # create a 0D tensor (scalar) from a Python integer
    tensor0d = torch.tensor(1)

    # create a 1D tensor (vector) from a Python list
    tensor1d = torch.tensor([1, 2, 3])

    # create a 2D tensor from a nested Python list
    tensor2d = torch.tensor([[1, 2], [3, 4]])

    # create a 3D tensor from a nested Python list
    tensor3d = torch.tensor([[[1, 2], [3, 4]], [[5, 6], [7, 8]]])

    print(tensor0d, tensor1d, tensor2d, tensor3d, sep="\n")
    return (tensor1d,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ##Tensor data types
    1. PyTorch adopts the default 64-bit integer data type from Python
    2. Python floats creates tensors with a 32-bit precision by default
    """)
    return


@app.cell
def _(tensor1d):
    # Tensor data type
    print(tensor1d.dtype)
    return


@app.cell
def _(torch):
    floatvec = torch.tensor([1.0, 2.0, 3.0])
    print(floatvec.dtype)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    It is possible to readily change the precision using a tensor’s .to method.
    """)
    return


@app.cell
def _(tensor1d, torch):
    floatvec2 = tensor1d.to(torch.float32)
    print(floatvec2.dtype)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ##Common PyTorch tensor operations

    Coverage of essential PyTorch tensor operations and commands
    """)
    return


@app.cell
def _(torch):
    tens2d = torch.tensor([[1, 2, 3],
                             [4, 5, 6]])
    tens2d
    return (tens2d,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    In addition, the .shape attribute allows us to access the shape of a tensor
    """)
    return


@app.cell
def _(tens2d):
    tens2d.shape
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    We can use .reshape to change the shape
    """)
    return


@app.cell
def _(tens2d):
    tens2d.reshape(3,2)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    The more common cmd for reshaping tensors is .view()
    """)
    return


@app.cell
def _(tens2d):
    tens2d.view(3,2)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Use .T for transposing the matrix
    """)
    return


@app.cell
def _(tens2d):
    tens2d.T
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Now the most important matrix multiplication can be done via .matmul or @ operator
    """)
    return


@app.cell
def _(tens2d):
    tens2d.matmul(tens2d.T)
    return


@app.cell
def _(tens2d):
    tens2d @ tens2d.T
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    #Seeing models as computation graphs

    PyTorch’s automatic differentiation engine, also known as autograd.

    ### Computational Graph

    A computational graph (or computation graph in short) is a directed graph that allows us to express and visualize mathematical expressions. In the context of deep learning, a computation graph lays out the sequence of calculations needed to compute the output of a neural network
    """)
    return


@app.cell
def _(torch):
    #Example

    import torch.nn.functional as F

    y = torch.tensor([1.0])  # true label
    x1 = torch.tensor([1.1]) # input feature
    w1 = torch.tensor([2.2], requires_grad=True) # weight parameter
    b = torch.tensor([0.0], requires_grad=True)  # bias unit

    z = x1 * w1 + b          # net input
    a = torch.sigmoid(z)     # activation & output

    loss = F.binary_cross_entropy(a, y)
    print(loss)
    return F, b, loss, w1


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    If we carry out computations in PyTorch, it will build a computational graph internally by default if one of its terminal nodes has the requires_grad attribute set to True. This is useful if we want to compute gradients. Gradients are required when training neural networks via the popular backpropagation algorithm, which can be thought of as an implementation of the chain rule from calculus for neural networks
    """)
    return


@app.cell
def _(b, loss, w1):
    from torch.autograd import grad

    grad_L_w1 = grad(loss, w1, retain_graph=True)
    grad_L_b = grad(loss, b, retain_graph=True)
    return grad_L_b, grad_L_w1


@app.cell
def _(grad_L_w1):
    grad_L_w1
    return


@app.cell
def _(grad_L_b):
    grad_L_b
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    we have been using the grad function “manually,” which can be useful for experimentation, debugging, and demonstrating concepts. But in practice, PyTorch provides even more high-level tools to automate this process. For instance, we can call .backward on the loss, and PyTorch will compute the gradients of all the leaf nodes in the graph, which will be stored via the tensors’ .grad attributes
    """)
    return


@app.cell
def _(b, loss, w1):
    loss.backward()

    print(w1.grad)
    print(b.grad)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Implementing Multilayer Neural Network

    When implementing a neural network in PyTorch, we typically subclass the torch.nn.Module class to define our own custom network architecture. This Module base class provides a lot of functionality, making it easier to build and train models. For instance, it allows us to encapsulate layers and operations and keep track of the model’s parameters.


    Within this subclass, we define the network layers in the __init__ constructor and specify how they interact in the forward method. The forward method describes how the input data passes through the network and comes together as a computation graph.
    """)
    return


@app.cell
def _(torch):
    #MLP example (multilayer perceptron)

    class NeuralNetwork(torch.nn.Module):
        def __init__(self, num_inputs, num_outputs):
            super().__init__()

            self.layers = torch.nn.Sequential(

                #1st Hidden Layer
                torch.nn.Linear(num_inputs,30),
                torch.nn.ReLU(),

                #2nd Hidden Layer
                torch.nn.Linear(30, 20),
                torch.nn.ReLU(),

                #Output Layer
                torch.nn.Linear(20,num_outputs),
            )

        def forward(self, x):
            logits = self.layers(x)
            return logits


    return (NeuralNetwork,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Why super.__init__()

    Our class inherits from torch.nn.Module which means our class is the child class and torch.nn.Module is the parent class. When we create an object (model = NeuralNetwork(2, 3)), it first calls our init method and then calls the super init (parent's init)

    ### Why is this necessary?

    torch.nn.Module performs a lot of important setup behind the scenes:

    1. Creates internal dictionaries to store layers (_modules)
    2. Registers trainable parameters (_parameters)
    3. Sets up buffers (_buffers)
    4. Enables methods like model.parameters(), model.to(device), model.train(), model.eval(), model.state_dict(), model.load_state_dict()

    Without calling it, PyTorch doesn't know your object is a proper neural network module.
    """)
    return


@app.cell
def _(NeuralNetwork):
    model = NeuralNetwork(60,5)
    return (model,)


@app.cell
def _(model):
    print(model)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    We can also check the num of trainable parameters
    """)
    return


@app.cell
def _(model):
    num_params = sum(
        p.numel() for p in model.parameters() if p.requires_grad
    )
    print("Total number of trainable model parameters:", num_params)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Based on the print(model) call we executed above, we can see that the first Linear layer is at index position 0 in the layers attribute. We can access the corresponding weight parameter matrix as follows:
    """)
    return


@app.cell
def _(model):
    print(model.layers[0].weight)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    As this matrix is large, lets try to print the shape of it
    """)
    return


@app.cell
def _(model):
    print(model.layers[0].weight.shape)
    return


@app.cell
def _(model, torch):
    #Lets try an example

    X = torch.rand((1,60))
    out = model(X)
    print(out)
    return (X,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    If we just want to use a network without training or backpropagation, for example, if we use it for prediction after training, constructing this computational graph for backpropagation can be wasteful as it performs unnecessary computations and consumes additional memory. So, when we use a model for inference (for instance, making predictions) rather than training, it is a best practice to use the torch.no_grad() context manager. This tells PyTorch that it doesn’t need to keep track of the gradients, which can result in significant savings in memory and computation.
    """)
    return


@app.cell
def _(X, model, torch):
    with torch.no_grad():
        out_no_grad = model(X)
    print(out_no_grad)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    In PyTorch, it’s common practice to code models such that they return the outputs of the last layer (logits) without passing them to a nonlinear activation function. That’s because PyTorch’s commonly used loss functions combine the softmax (or sigmoid for binary classification) operation with the negative log-likelihood loss in a single class. The reason for this is numerical efficiency and stability. So, if we want to compute class-membership probabilities for our predictions, we have to call the softmax function explicitly
    """)
    return


@app.cell
def _(X, model, torch):
    with torch.no_grad():
        out_soft = torch.softmax(model(X), dim=1)
    print(out_soft)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Setting up efficient data loaders

    PyTorch implements a `Dataset` and a `DataLoader` class. The `Dataset` class is used to instantiate objects that define how each data record is loaded. The `DataLoader` handles how the data is shuffled and assembled into batches
    """)
    return


@app.cell
def _(torch):
    #Example

    X_train = torch.tensor([
        [-1.2, 3.1],
        [-0.9, 2.9],
        [-0.5, 2.6],
        [2.3, -1.1],
        [2.7, -1.5]
    ])

    return (X_train,)


@app.cell
def _(torch):
    y_train = torch.tensor([0, 0, 0, 1, 1])
    return (y_train,)


@app.cell
def _(torch):
    #Test set

    X_test = torch.tensor([
        [-0.8, 2.8],
        [2.6, -1.6],
    ])

    y_test = torch.tensor([0, 1])
    return X_test, y_test


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    we create a custom dataset class, ToyDataset, by subclassing from PyTorch’s Dataset parent class
    """)
    return


@app.cell
def _(X_test, X_train, y_test, y_train):
    from torch.utils.data import Dataset

    class ToyDataset(Dataset):
        def __init__(self, X, y):
            self.features = X
            self.labels = y

        def __getitem__(self, index):
            one_x = self.features[index]
            one_y = self.labels[index]
            return one_x, one_y

        def __len__(self):
            return self.labels.shape[0]

    train_ds = ToyDataset(X_train, y_train)
    test_ds = ToyDataset(X_test, y_test)
    return test_ds, train_ds


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    In PyTorch, the three main components of a custom Dataset class are the __init__ constructor, the __getitem__ method, and the __len__ method
    """)
    return


@app.cell
def _(train_ds):
    len(train_ds)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Now that we defined a PyTorch Dataset class we can use for our toy dataset, we can use PyTorch’s DataLoader class to sample from it
    """)
    return


@app.cell
def _(torch, train_ds):
    from torch.utils.data import DataLoader

    torch.manual_seed(42)

    train_loader = DataLoader(
        dataset=train_ds,
        batch_size=2,
        shuffle=True,
        num_workers=0,
        drop_last=True #to avoid the last incomplete batch
    )
    return DataLoader, train_loader


@app.cell
def _(DataLoader, test_ds):
    test_loader = DataLoader(
        dataset=test_ds,
        batch_size=2,
        shuffle=False,
        num_workers=0
    )
    return


@app.cell
def _(train_loader):
    #After instantiating the training data loader, we can iterate over it

    for idx, (x, y1) in enumerate(train_loader):
        print(f"Batch {idx+1}:", x, y1)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    #One full Training loop
    """)
    return


@app.cell
def _(F, NeuralNetwork, torch, train_loader):
    torch.manual_seed(123)

    model1 = NeuralNetwork(2,2)

    optim = torch.optim.SGD(model1.parameters(), lr=0.5)

    num_epochs = 3

    for epoch in range(num_epochs):
        model1.train()
        for batch_idx, (features, labels) in enumerate(train_loader):
            logits1 = model1(features)
            loss1 = F.cross_entropy(logits1, labels) #loss function

            optim.zero_grad() # to update the gradients to zero
            loss1.backward() #backpropagation
            optim.step() #update the weights

            ##Logging
            print(f"Epoch: {epoch+1:03d}/{num_epochs:03d}"
                  f" | Batch {batch_idx:03d}/{len(train_loader):03d}"
                  f" | Train/Val Loss: {loss1:.2f}")

            model1.eval()
    return (model1,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    We can use torch.no_grad() now to use it for inferencing
    """)
    return


@app.cell
def _(X_test, model1, torch):
    with torch.no_grad():
        outputs = model1(X_test)

    print(outputs)
    return (outputs,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    To obtain the class membership probabilities, we can then use PyTorch’s softmax function
    """)
    return


@app.cell
def _(outputs, torch):
    torch.set_printoptions(sci_mode=False)

    probabs = torch.softmax(outputs, dim=1)
    print(probabs)
    return (probabs,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    We can convert these values into class labels predictions using PyTorch’s argmax function, which returns the index position of the highest value in each row if we set dim=1 (setting dim=0 would return the highest value in each column, instead):
    """)
    return


@app.cell
def _(probabs, torch):
    predictions = torch.argmax(probabs, dim=1)
    print(predictions)
    return (predictions,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    It is unnecessary to compute softmax probabilities to obtain the class labels. We could also apply the argmax function to the logits (outputs) directly
    """)
    return


@app.cell
def _(outputs, torch):
    predictions1 = torch.argmax(outputs, dim=1)
    print(predictions1)
    return


@app.cell
def _(predictions, y_test):
    predictions == y_test
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Using torch.sum, we can count the number of correct prediction
    """)
    return


@app.cell
def _(predictions, torch, y_test):
    torch.sum(predictions == y_test)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    To generalize the computation of the prediction accuracy, let’s implement a compute_accuracy function
    """)
    return


@app.cell
def _(model1, torch):
    def compute_accuracy(model, dataloader):
        model.eval()
        correct = 0.0
        total_examples = 0

        for idx , (features, labels) in enumerate(dataloader):
            with torch.no_grad():
                output1 = model1(features)

            preds = torch.argmax(output1, dim=1)
            compare = labels == preds
            correct+= torch.sum(compare)
            total_examples += len(compare)

        return (correct/total_examples).item()


    return (compute_accuracy,)


@app.cell
def _(compute_accuracy, model1, train_loader):
    compute_accuracy(model1, train_loader)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    #Saving and loading models
    """)
    return


@app.cell
def _(model1, torch):
    torch.save(model1.state_dict(), "model1.pth")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    The model’s state_dict is a Python dictionary object that maps each layer in the model to its trainable parameters (weights and biases). Note that "model.pth" is an arbitrary filename for the model file saved to disk. We can give it any name and file ending we like; however, .pth and .pt are the most common conventions.

    Once we saved the model, we can restore it from disk also
    """)
    return


@app.cell
def _(NeuralNetwork, torch):
    model_old = NeuralNetwork(2,2)
    model_old.load_state_dict(torch.load("model1.pth", weights_only=True))
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    #Optimizing training performance with GPUs
    """)
    return


@app.cell
def _(torch):
    torch.backends.mps.is_available()
    return


@app.cell
def _(torch):
    tensor_1 = torch.tensor([1., 2., 3.])
    tensor_2 = torch.tensor([4., 5., 6.])

    print(tensor_1 + tensor_2)
    return tensor_1, tensor_2


@app.cell
def _(tensor_1, tensor_2):
    tensor_1_gpu = tensor_1.to("mps")
    tensor_2_gpu = tensor_2.to("mps")

    print(tensor_1_gpu + tensor_2_gpu)
    return (tensor_2_gpu,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    All tensors must be on the same device. Otherwise, the computation will fail, as shown below, where one tensor resides on the CPU and the other on the GPU
    """)
    return


@app.cell
def _(tensor_1, tensor_2_gpu):
    #will give error
    print(tensor_1 + tensor_2_gpu)
    return


@app.cell
def _():
    return


if __name__ == "__main__":
    app.run()
