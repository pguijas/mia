using DelimitedFiles
using Random
include("utils.jl")

Random.seed!(1)

# Load the dataset
println("Loading data...")
dataset = readdlm("iris.data",',')

# Split information into inputs and targets
inputs = normalizeZeroMean(Float32.(dataset[:,1:4]))
targets = String.(dataset[:,5])

println("Size of inputs: $(size(inputs))")
println("Size of targets: $(size(targets))")
@assert (size(inputs, 1) == size(targets, 1)) "There are different number of samples in inputs and targets"

# Create the cross-validation indexes
k = 10
crossValidationIdxs = crossvalidation(targets, k)

# Set the hyperparameters
hyperparameters = Dict(
    # ANN
    "topology" => [8],
    "maxEpochs" => 1000,
    "minLoss" => 0.0,
    "learningRate" => 0.01,
    "repetitionsTraining" => 50,
    "validationRatio" => 0.1,
    "maxEpochsVal" => 25,
    
    # SVM
    "kernel" => "linear",
    "degree" => 3,
    "gamma" => 2,
    "C" => 1,
    
    # Decision Tree
    "depth" => 4,
    
    # kNN
    "numNeighbours" => 5
    )

# Train each model individually
println("\nTraining the models individually...")
println("ANN: ", modelCrossValidation(:ANN, hyperparameters, inputs, targets, crossValidationIdxs))
println("SVM: ", modelCrossValidation(:SVM, hyperparameters, inputs, targets, crossValidationIdxs))
println("Decision Tree: ", modelCrossValidation(:DecisionTree, hyperparameters, inputs, targets, crossValidationIdxs))
println("kNN: ", modelCrossValidation(:kNN, hyperparameters, inputs, targets, crossValidationIdxs))

# Train a model with the full dataset and print the confusion matrix
println("\nTraining the best model configuration...")
trainModelFullDataset(:kNN, hyperparameters, (inputs, targets));

# Train an ensemble of the 4 models
estimators = [:ANN, :SVM, :DecisionTree, :kNN]
println("\nTraining an ensemble of the 4 models...")
println(trainClassEnsemble(estimators, hyperparameters, (inputs, targets), crossValidationIdxs))

# Train an ensemble of 50 SVM
hyperparameters = Dict(
    "kernel" => "linear",
    "degree" => 3,
    "gamma" => 2,
    "C" => 1
)
println("\nTraining an ensemble of 50 SVM...")
println(trainClassEnsemble(:SVM, hyperparameters, (inputs, targets), crossValidationIdxs, 50))
