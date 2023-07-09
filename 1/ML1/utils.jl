using Flux
using Flux.Losses

using Random

using ScikitLearn
@sk_import svm: SVC
@sk_import tree: DecisionTreeClassifier
@sk_import neighbors: KNeighborsClassifier
@sk_import neural_network:MLPClassifier
@sk_import ensemble:VotingClassifier

using Statistics

##############################################
# Unit 2: output to boolean transformations
##############################################

function classifyOutputs(outputs::AbstractArray{<:Real,2}; threshold::Real=0.5) 
    nOutputs = size(outputs, 2)
    if nOutputs == 1
        outputs = (outputs .>= threshold)
    else
        (_, indicesMaxEachInstance) = findmax(outputs, dims=2)
        outputs = falses(size(outputs))
        outputs[indicesMaxEachInstance] .= true;
    end
    outputs = convert(Array{Bool,2}, outputs)
    return outputs
end;

function oneHotEncoding(feature::AbstractArray{<:Any,1}, classes::AbstractArray{<:Any,1})
    nClasses = length(classes)
    if nClasses == 2
        feat = convert(Array{Bool}, feature .== classes[1])  # encode the first class as 1/true and the second class as 0/false
        feat = reshape(feat, (length(feat), 1))  # convert from Vector to Matrix with one column
    else
        # permutedims is necessary because the transpose operation is not defined for e.g. string vectors
        feat = Array{Bool,2}((permutedims(feature) .== classes)')
    end
    return feat
end;

oneHotEncoding(feature::AbstractArray{<:Any,1}) = oneHotEncoding(feature, unique(feature));
#oneHotEncoding(feature::AbstractArray{Bool,1}) = oneHotEncoding(feature, unique(feature));  # call oneHotEncoding, more cleaner code


##############################################
# Unit 2: dataset normalization
##############################################

# MinMax normalization
function calculateMinMaxNormalizationParameters(dataset::AbstractArray{<:Real,2})
    return minimum(dataset, dims=1), maximum(dataset, dims=1)
end;

function normalizeMinMax!(dataset::AbstractArray{<:Real,2},      
    normalizationParameters::NTuple{2, AbstractArray{<:Real,2}})
    min, max = normalizationParameters
    dataset .-= min
    dataset ./= (max .- min)
    dataset[:, vec(max .== min)] .= 0  # can be removed the attributes whose max == min in the same variable? (no reassign)
    return dataset
end;

function normalizeMinMax!(dataset::AbstractArray{<:Real,2})
    min, max = calculateMinMaxNormalizationParameters(dataset)
    return normalizeMinMax!(dataset, (min, max))
end;

function normalizeMinMax(dataset::AbstractArray{<:Real,2},      
    normalizationParameters::NTuple{2, AbstractArray{<:Real,2}}) 
    newDataset = copy(dataset)
    normalizeMinMax!(newDataset, normalizationParameters)
    return newDataset
end;

function normalizeMinMax(dataset::AbstractArray{<:Real,2})
    return normalizeMinMax(copy(dataset), calculateMinMaxNormalizationParameters(dataset))
end;

# ZeroMean normalization
function calculateZeroMeanNormalizationParameters(dataset::AbstractArray{<:Real,2})
    return mean(dataset, dims=1), std(dataset, dims=1)
end;

function normalizeZeroMean!(dataset::AbstractArray{<:Real,2},      
    normalizationParameters::NTuple{2, AbstractArray{<:Real,2}}) 
    mean, std = normalizationParameters
    dataset .-= mean
    dataset ./= std
    dataset[:, vec(std .== 0)] .= 0  # can be removed the attributes whose std == 0 in the same variable? (no reassign)
    return dataset
end; 

function normalizeZeroMean!(dataset::AbstractArray{<:Real,2})
    mean, std = calculateZeroMeanNormalizationParameters(dataset)
    return normalizeZeroMean!(dataset, (mean, std))
end;

function normalizeZeroMean(dataset::AbstractArray{<:Real,2},      
    normalizationParameters::NTuple{2, AbstractArray{<:Real,2}})
    newDataset = copy(dataset)
    normalizeZeroMean!(newDataset, normalizationParameters)
    return newDataset
end;

function normalizeZeroMean(dataset::AbstractArray{<:Real,2}) 
    return normalizeZeroMean(dataset, calculateZeroMeanNormalizationParameters(dataset))
end;


function accuracy(outputs::AbstractArray{Bool,1}, targets::AbstractArray{Bool,1}) 
    return mean(outputs .== targets)
end;


##############################################
# Unit 2,4: metrics calculation
##############################################

function accuracy(outputs::AbstractArray{Bool,1}, targets::AbstractArray{Bool,1}) 
    return mean(outputs .== targets)
end;

function accuracy(outputs::AbstractArray{Bool,2}, targets::AbstractArray{Bool,2}) 
    nOutputs = size(outputs, 2)
    @assert nOutputs != 2 "The number of outputs cannot be 2"
    
    if nOutputs == 1
        acc = accuracy(outputs[:,1], targets[:,1])
    else
        classComparison = targets .!= outputs 
        incorrectClassifications = any(classComparison, dims=2)
        acc = 1 - mean(incorrectClassifications)
    end
    return acc
end;

function accuracy(outputs::AbstractArray{<:Real,1}, targets::AbstractArray{Bool,1}; threshold::Real=0.5)
    return accuracy(outputs .>= threshold, targets)
end;

function accuracy(outputs::AbstractArray{<:Real,2}, targets::AbstractArray{Bool,2}; threshold::Real=0.5)
    nOutputs = size(outputs, 2)
    @assert nOutputs != 2 "The number of outputs cannot be 2"

    if nOutputs == 1
        return accuracy(outputs[:,1], targets[:,1], threshold=threshold)
    else 
        return accuracy(classifyOutputs(outputs), targets)
    end
end;

function confusionMatrix(outputs::AbstractArray{Bool,1}, targets::AbstractArray{Bool,1})
    
    # Build the confusion matrix
    #  ---------
    # | TN | FP |
    # |---------|
    # | FN | TP |
    #  ---------
    tn = sum(.!outputs .* .!targets)
    fn = sum(.!outputs .* targets)
    fp = sum(outputs .* .!targets)
    tp = sum(outputs .* targets)    
    matrix = [tn fp; fn tp]
    
    # Calculate the metrics
    acc = (tn + tp) / (tn + tp + fn + fp)
    errorRate = 1 - acc
    sensitivity = tp / (fn + tp)
    specificity = tn / (fp + tn)
    ppv = tp / (tp + fp)
    npv = tn / (tn + fn)
    
    # Correct metrics when all patterns are TN
    if isnan(sensitivity) && isnan(ppv)
        sensitivity = 1.0
        ppv = 1.0
    end
    
    # Correct metrics when all patterns are TP
    if isnan(specificity) && isnan(npv)
        specificity = 1.0
        npv = 1.0
    end
    
    # Correct metrics that cannot be calculated in other cases
    if isnan(sensitivity)
        sensitivity = 0.0
    end
    
    if isnan(specificity)
        specificity = 0.0
    end
    
    if isnan(ppv)
        ppv = 0.0
    end
    
    if isnan(npv)
        npv = 0.0
    end

    # Calculate f1-score
    if (sensitivity == 0.0) && (ppv == 0.0)
        f1Score = 0.0
    else
        f1Score = 2 * (sensitivity * ppv) / (sensitivity + ppv)
    end
    
    return acc, errorRate, sensitivity, specificity, ppv, npv, f1Score, matrix
end;

function confusionMatrix(outputs::AbstractArray{<:Real,1},targets::AbstractArray{Bool,1}; threshold::Real=0.5)
    return confusionMatrix(outputs .>= threshold, targets)
end;

function printConfusionMatrix(outputs::AbstractArray{Bool,1},targets::AbstractArray{Bool,1})
    acc, errorRate, sensitivity, specificity, ppv, npv, f1Score, matrix = confusionMatrix(outputs, targets)
    
    println("accuracy=$(acc), error rate=$(errorRate), sensitivity=$(sensitivity), specificity=$(specificity), ppv=$(ppv), npv=$(npv), f1-score=$(f1Score)")
    
    println("\t \t-\t \t+\t")
    println("\t|---------------|---------------|")
    print("-\t|")
    print.("\t", matrix[1,:], "\t|")
    println("")
    println("\t|---------------|---------------|")
    print("+\t|")
    print.("\t", matrix[2,:], "\t|")
    println("")
    println("\t|---------------|---------------|")
end;

function printConfusionMatrix(outputs::AbstractArray{<:Real,1},targets::AbstractArray{Bool,1}; threshold::Real=0.5)
    printConfusionMatrix(outputs .>= threshold, targets)
end;

function confusionMatrix(outputs::AbstractArray{Bool,2}, targets::AbstractArray{Bool,2}; weighted::Bool=true)
    
    # Check the number of classes is correct and both outputs and targets have the same
    numClasses = size(outputs, 2)
    @assert (numClasses == size(targets,2)) "outputs and targets have different number of classes"
    @assert (numClasses != 2) "Cannot exist outputs or targets with 2 columns"
    
    if (numClasses == 1)
        return confusionMatrix(outputs[:,1], targets[:,1])
    end
    
    # Calculate metrics
    sensitivity = zeros(numClasses)
    specificity = zeros(numClasses)
    ppv = zeros(numClasses)
    npv = zeros(numClasses)
    f1Score = zeros(numClasses)
    
    numInstances = sum(targets, dims=1)
    for numClass in 1:numClasses
        if (numInstances[numClasses] == 0)
            continue
        end
        _, _, sensitivity[numClass], specificity[numClass], ppv[numClass], npv[numClass], f1Score[numClass], _ = confusionMatrix(outputs[:,numClass], targets[:,numClass]);
    end
    
    # Fill confusion matrix
    matrix = zeros(numClasses, numClasses)
    for numClassTarget in 1:numClasses, numClassOutput in 1:numClasses
        matrix[numClassTarget, numClassOutput] = sum(targets[:, numClassTarget] .* outputs[:, numClassOutput])
    end
    
    # Aggregate metrics according to the strategy specified
    if (weighted)
        weightClasses = numInstances ./ sum(numInstances)
        sensitivity = sum(sensitivity .* weightClasses)
        specificity = sum(specificity .* weightClasses)
        ppv = sum(ppv .* weightClasses)
        npv = sum(npv .* weightClasses)
        f1Score = sum(f1Score .* weightClasses)
    else
        sensitivity = mean(sensitivity)
        specificity = mean(specificity)
        ppv = mean(ppv)
        npv = mean(npv)
        f1Score = mean(f1Score)
    end
    
    acc = accuracy(outputs, targets)
    errorRate = 1 - acc
    
    return acc, errorRate, sensitivity, specificity, ppv, npv, f1Score, matrix
end;

function confusionMatrix(outputs::AbstractArray{<:Real,2}, targets::AbstractArray{Bool,2}; weighted::Bool=true)
    return confusionMatrix(classifyOutputs(outputs), targets; weighted=weighted)
end;

function confusionMatrix(outputs::AbstractArray{<:Any,1}, targets::AbstractArray{<:Any,1}; weighted::Bool=true)
    @assert (all([in(output, unique(targets)) for output in outputs])) "targets does not contain all the classes in outputs"
    
    classes = unique(targets);
    return confusionMatrix(oneHotEncoding(outputs, classes), oneHotEncoding(targets, classes); weighted=weighted);
end;


# Extra functions to print multiclass confusion matrices (adapted from the AA course)
function printConfusionMatrix(outputs::Array{Bool,2}, targets::Array{Bool,2}; weighted::Bool=true)
    acc, errorRate, sensitivity, specificity, ppv, npv, f1Score, matrix = confusionMatrix(outputs, targets; weighted=weighted);
    
    println("accuracy=$(acc), error rate=$(errorRate), sensitivity=$(sensitivity), specificity=$(specificity), ppv=$(ppv), npv=$(npv), f1-score=$(f1Score)")
    
    numClasses = size(matrix, 1)
    writeHorizontalLine() = (for i in 1:numClasses+1 print("--------") end; println(""); );
    writeHorizontalLine();
    print("\t| ");
    if (numClasses==2)
        println(" - \t + \t|");
    else
        print.("Cl. ", 1:numClasses, "\t| ");
    end;
    println("");
    writeHorizontalLine();
    for numClassTarget in 1:numClasses
        if (numClasses==2)
            print(numClassTarget == 1 ? " - \t| " : " + \t| ");
        else
            print("Cl. ", numClassTarget, "\t| ");
        end;
        print.(matrix[numClassTarget,:], "\t| ");
        println("");
        writeHorizontalLine();
    end
end;

printConfusionMatrix(outputs::Array{<:Real,2}, targets::Array{Bool,2}; weighted::Bool=true) =
    printConfusionMatrix(classifyOutputs(outputs), targets; weighted=weighted);

function printConfusionMatrix(outputs::Array{<:Any,1}, targets::Array{<:Any,1}; weighted::Bool=true)
    @assert (all([in(output, unique(targets)) for output in outputs])) "targets does not contain all the classes in outputs"
    
    classes = unique(targets)
    printConfusionMatrix(oneHotEncoding(outputs, classes), oneHotEncoding(targets, classes); weighted=weighted)
end;


##############################################
# Unit 2,3,5: build and train an ANN
##############################################

function buildClassANN(numInputs::Int, topology::AbstractArray{<:Int,1}, numOutputs::Int; transferFunctions::AbstractArray{<:Function,1}=fill(σ, length(topology))) 
    ann = Chain();
    numInputsLayer = numInputs
    
    # Hidden layers
    for i in 1:length(topology)
        numOutputsLayer = topology[i]
        transferFunction = transferFunctions[i]
        ann = Chain(ann..., Dense(numInputsLayer, numOutputsLayer, transferFunction))      
        numInputsLayer = numOutputsLayer
    end
    
    # Output layer
    if numOutputs == 1
        ann = Chain(ann..., Dense(numInputsLayer, numOutputs, σ))
    else
        ann = Chain(ann..., Dense(numInputsLayer, numOutputs, identity))
        ann = Chain(ann..., softmax)
    end
    
    return ann
end;

function trainClassANN(topology::AbstractArray{<:Int,1},  
            trainingDataset::Tuple{AbstractArray{<:Real,2}, AbstractArray{Bool,2}}; 
            validationDataset::Tuple{AbstractArray{<:Real,2}, AbstractArray{Bool,2}}= 
                    (Array{eltype(trainingDataset[1]),2}(undef,0,0), falses(0,0)), 
            testDataset::Tuple{AbstractArray{<:Real,2}, AbstractArray{Bool,2}}= 
                    (Array{eltype(trainingDataset[1]),2}(undef,0,0), falses(0,0)), 
            transferFunctions::AbstractArray{<:Function,1}=fill(σ, length(topology)), 
            maxEpochs::Int=1000, minLoss::Real=0.0, learningRate::Real=0.01,  
            maxEpochsVal::Int=20, showText::Bool=false)
    
    # Split into inputs/targets
    trainingInputs, trainingTargets = trainingDataset
    validationInputs, validationTargets = validationDataset
    testInputs, testTargets = testDataset
    
    useValidation = length(validationInputs) > 0
   
    # Check that the targets corresponding to the inputs have the same number of samples
    @assert (size(trainingInputs, 1) == size(trainingTargets, 1)) "Number of training inputs and targets do not match"
    @assert (size(testInputs, 1) == size(testTargets, 1)) "Number of test inputs and targets do not match"
    
    # Check that all the subsets have the same number of attributes/classes
    @assert (size(trainingTargets, 2) == size(testTargets, 2)) "Number of classes for training and test do not match"
    
    if (useValidation)
        @assert (size(validationInputs, 1) == size(validationTargets, 1)) "Number of validation inputs and targets do not match"
        @assert (size(trainingTargets, 2) == size(validationTargets, 2)) "Number of classes for training, validation and test do not match"
        @assert (size(trainingInputs, 2) == size(validationInputs, 2)) "Number of attributes for training and validation do not match"
    end
    if (size(testInputs, 1) != 0)
        @assert (size(trainingInputs, 2) == size(testInputs, 2)) "Number of attributes for training and test do not match"
    end
   
    # Build the network
    nInputs, nOutputs = size(trainingInputs, 2), size(trainingTargets, 2)
    ann = buildClassANN(nInputs, topology, nOutputs; transferFunctions)
    if (showText)
        println("ANN network built: $ann")
    end
    
    # Loss
    loss(x,y) = (size(y, 1) == 1) ? Losses.binarycrossentropy(ann(x), y) : Losses.crossentropy(ann(x), y)
    
    # Metric progress
    trainingLosses = Array{Float32}(undef, 0)
    validationLosses = Array{Float32}(undef, 0)
    testLosses = Array{Float32}(undef, 0)
    trainingAccs = Array{Float32}(undef, 0)
    validationAccs = Array{Float32}(undef, 0)
    testAccs = Array{Float32}(undef, 0)

    # Train for n=maxEpochs (at most) epochs
    currentEpoch = 0

    # Calculate, store and print last loss/accuracy
    function calculateMetrics()
        
        # Losses
        trainingLoss   = loss(trainingInputs', trainingTargets')
        validationLoss = (size(validationInputs, 1) != 0) ? loss(validationInputs', validationTargets') : 0
        testLoss       = (size(testInputs, 1) != 0) ? loss(testInputs', testTargets') : 0
     
        # Accuracies
        trainingOutputs   = ann(trainingInputs')
        validationOutputs = (size(validationInputs, 1) != 0) ? ann(validationInputs') : 0
        testOutputs       = ann(testInputs')
        trainingAcc   = accuracy(trainingOutputs',   trainingTargets)
        validationAcc = (size(validationInputs, 1) != 0) ? accuracy(validationOutputs', validationTargets) : 0
        testAcc       = accuracy(testOutputs',       testTargets)

        # Update the history of losses and accuracies
        push!(trainingLosses, trainingLoss)
        push!(validationLosses, validationLoss)
        push!(testLosses, testLoss)
        push!(trainingAccs, trainingAcc)
        push!(validationAccs, validationAcc)
        push!(testAccs, testAcc)
            
        # Show text
        if showText && (currentEpoch % 50 == 0)
            println("Epoch ", currentEpoch, 
                ": \n\tTraining loss: ", trainingLoss, ", accuracy: ", 100 * trainingAcc, 
                "% \n\tValidation loss: ", validationLoss, ", accuracy: ", 100 * validationAcc, 
                "% \n\tTest loss: ", testLoss, ", accuracy: ", 100 * testAcc, "%")
        end
        
        return trainingLoss, trainingAcc, validationLoss, validationAcc, testLoss, testAcc
    end

    # Compute and store initial metrics
    trainingLoss, _, validationLoss, _, _, _ = calculateMetrics()

    # Best model at validation set 
    numEpochsValidation = 0
    bestValidationLoss = validationLoss
    
    if (useValidation)
        bestANN = deepcopy(ann)
    else
        bestANN = ann  # if no validation, we want to return the ANN that is trained in every cycle
    end
    
    # Start the training
    while (currentEpoch < maxEpochs) && (trainingLoss > minLoss) && (numEpochsValidation < maxEpochsVal)
            
        # Update epoch number
        currentEpoch += 1
        
        # Fit the model
        Flux.train!(loss, Flux.params(ann), [(trainingInputs', trainingTargets')], ADAM(learningRate))
     
        # Compute and store metrics
        trainingLoss, _, validationLoss, _, _, _ = calculateMetrics()

        # Update validation early stopping only if validation set given
        if (useValidation)    
            if (validationLoss < bestValidationLoss)
                bestValidationLoss = validationLoss
                numEpochsValidation = 0
                bestANN = deepcopy(ann)
            else
                numEpochsValidation += 1
            end
        end
    end
    
    # Print stop reason and final metrics
    if (showText)
        println("Final results for epoch ", currentEpoch, 
                ": \n\tTraining loss: ", trainingLosses[end], ", accuracy: ", 100 * trainingAccs[end], 
                "% \n\tValidation loss: ", validationLosses[end], ", accuracy: ", 100 * validationAccs[end], 
                "% \n\tTest loss: ", testLosses[end], ", accuracy: ", 100 * testAccs[end], "%")

        println("\nStopping criteria: ",
            "\n\tmaxEpochs: ", !(currentEpoch < maxEpochs),
            "\n\tminLoss: ", !(trainingLoss > minLoss),
            "\n\tnumEpochsValidation: ", !(numEpochsValidation < maxEpochsVal))
    end
    
    return bestANN, trainingLosses, validationLosses, testLosses, trainingAccs, validationAccs, testAccs
end;

function trainClassANN(topology::AbstractArray{<:Int,1},  
        trainingDataset::Tuple{AbstractArray{<:Real,2}, AbstractArray{Bool,1}}; 
        validationDataset::Tuple{AbstractArray{<:Real,2}, AbstractArray{Bool,1}}= 
                    (Array{eltype(trainingDataset[1]),2}(undef,0,0), falses(0)), 
        testDataset::Tuple{AbstractArray{<:Real,2}, AbstractArray{Bool,1}}= 
                    (Array{eltype(trainingDataset[1]),2}(undef,0,0), falses(0)), 
        transferFunctions::AbstractArray{<:Function,1}=fill(σ, length(topology)), 
        maxEpochs::Int=1000, minLoss::Real=0.0, learningRate::Real=0.01,  
        maxEpochsVal::Int=20, showText::Bool=false)
    
    trainingInputs, trainingTargets = trainingDataset
    validationInputs, validationTargets = validationDataset
    testInputs, testTargets = testDataset
    trainingTargets = reshape(trainingTargets, (length(trainingTargets), 1))
    validationTargets = reshape(validationTargets, (length(validationTargets), 1))
    testTargets = reshape(testTargets, (length(testTargets), 1))
    
    return trainClassANN(topology, (trainingInputs, trainingTargets), validationDataset=(validationInputs, validationTargets),
        testDataset= (testInputs, testTargets), transferFunctions=transferFunctions, maxEpochs=maxEpochs, 
        minLoss=minLoss, learningRate=learningRate, maxEpochsVal=maxEpochsVal, showText=showText)
end;

function trainClassANN(topology::AbstractArray{<:Int,1}, 
        trainingDataset::Tuple{AbstractArray{<:Real,2}, AbstractArray{Bool,2}}, 
        kFoldIndices::	Array{Int64,1}; 
        transferFunctions::AbstractArray{<:Function,1}=fill(σ, length(topology)), 
        maxEpochs::Int=1000, minLoss::Real=0.0, learningRate::Real=0.01, repetitionsTraining::Int=1, 
        validationRatio::Real=0.0, maxEpochsVal::Int=20)
    
    inputs, targets = trainingDataset
    
    # crossvalidation variables
    k = maximum(kFoldIndices)
    testAccsK = zeros(k)
    
    # Train with k different splits
    for ki in 1:k
        
        # Use the patterns with no k index for train
        trainingInputs = inputs[kFoldIndices .!= ki, :]
        trainingTargets = targets[kFoldIndices .!= ki, :]
        
        # Split the training subset into train and validation
        trainingIdx, validationIdx = holdOut(size(trainingInputs, 1), validationRatio) 
        trainingInputs, validationInputs = trainingInputs[trainingIdx, :], trainingInputs[validationIdx, :]
        trainingTargets, validationTargets = trainingTargets[trainingIdx, :], trainingTargets[validationIdx, :]
        
        # Use the patterns with the k index for test
        testInputs = inputs[kFoldIndices .== ki, :]
        testTargets = targets[kFoldIndices .== ki, :]
        
        # Train each network several times and save the accuracy
        accs = zeros(repetitionsTraining)
        
        for i in 1:repetitionsTraining
            ann, trainingLosses, validationLosses, testLosses, trainingAccs, validationAccs, testAccs = trainClassANN(
                topology, (trainingInputs, trainingTargets), validationDataset=(validationInputs, validationTargets), 
                testDataset=(testInputs, testTargets), transferFunctions=transferFunctions, maxEpochs=maxEpochs, 
                minLoss=minLoss, learningRate=learningRate, maxEpochsVal=maxEpochsVal)

            # If using validation, the model may not correspond to the last epoch so we cannot just get
            # testAccs[end] as the accuracy of the model
            accs[i] = accuracy(ann(testInputs')', testTargets)
        end
            
        # Compute the mean of the results and save them
        testAccsK[ki] = mean(accs)
    end
    
    # Return the average and std of the metrics in the different k folds
    return mean(testAccsK), std(testAccsK)
end;

function trainClassANN(topology::AbstractArray{<:Int,1},
        trainingDataset::Tuple{AbstractArray{<:Real,2}, AbstractArray{Bool,1}},
        kFoldIndices::	Array{Int64,1};
        transferFunctions::AbstractArray{<:Function,1}=fill(σ, length(topology)),
        maxEpochs::Int=1000, minLoss::Real=0.0, learningRate::Real=0.01, repetitionsTraining::Int=1, 
        validationRatio::Real=0.0, maxEpochsVal::Int=20)
    
    trainingInputs, trainingTargets = trainingDataset
    trainingTargets = reshape(trainingTargets, (length(trainingTargets), 1))
    
    return trainClassANN(topology, (trainingInputs, trainingTargets), kFoldIndices, transferFunctions=transferFunctions, 
        maxEpochs=maxEpochs, minLoss=minLoss, learningRate=learningRate, repetitionsTraining=repetitionsTraining,
        validationRatio=validationRatio, maxEpochsVal=maxEpochsVal)
end;

##############################################
# Unit 3,5: hold-out and cross-validation
##############################################

function holdOut(N::Int, P::Real)
    @assert ((P >= 0.) & (P <= 1.)) "The percentage must be a number between 0 and 1"
    
    indices = randperm(N)
    sizeTrainSet = Int(round(N * (1 - P)))
    return indices[1:sizeTrainSet], indices[sizeTrainSet + 1:end]
end;

function holdOut(N::Int, Pval::Real, Ptest::Real) 
    @assert ((Pval + Ptest) <= 1.)  "The percentage for validation and test cannot be greater than 1"
    
    # Test indices
    (trainValIndices, testIndices) = holdOut(N, Ptest)
    # Train and Validation indices (Pval correction)
    (trainingIndices, validationIndices) = holdOut(length(trainValIndices), Pval * N / length(trainValIndices))
    @assert ((length(trainingIndices) + length(validationIndices) + length(testIndices)) == N)
    
    return trainValIndices[trainingIndices], trainValIndices[validationIndices], testIndices
end;

function crossvalidation(N::Int64, k::Int64)
    idx = repeat(1:k, Int32(ceil(N / k)))[1:N]
    shuffle!(idx)
    return idx
end;

function crossvalidation(targets::AbstractArray{Bool,2}, k::Int64)
    @assert all(sum(targets, dims=1) .>= k) "There are no enough instances per class to perform a $(k)-crossvalidation"
    
    idx = Int.(zeros(size(targets, 1)))
    for class in 1:size(targets, 2)
        idx[targets[:, class]] .= crossvalidation(sum(targets[:, class]), k)
    end
    
    return idx
end;

function crossvalidation(targets::AbstractArray{<:Any,1}, k::Int64)
    return crossvalidation(oneHotEncoding(targets), k)
end;

##############################################
# Unit 6, 7: Scikit-Learn models and ensembles
##############################################

function modelCrossValidation(modelType::Symbol,
        modelHyperparameters::Dict,
        inputs::AbstractArray{<:Real,2},
        targets::AbstractArray{<:Any,1},
        crossValidationIndices::Array{Int64,1})

    @assert (in(modelType, [:ANN, :SVM, :kNN, :DecisionTree])) "Model type $(modelType) is not supported"
    
    # Train an ANN model
    if (modelType == :ANN)
        targets = oneHotEncoding(targets)
        
        # change trainClassANN for more metrics
        # we use the default transfer functions (sigmoid)
        meanAcc, stdAcc = trainClassANN(modelHyperparameters["topology"], (inputs, targets), crossValidationIndices;
            maxEpochs=modelHyperparameters["maxEpochs"], minLoss=modelHyperparameters["minLoss"],
            learningRate=modelHyperparameters["learningRate"], repetitionsTraining=modelHyperparameters["repetitionsTraining"],
            validationRatio=modelHyperparameters["validationRatio"], maxEpochsVal=modelHyperparameters["maxEpochsVal"])
        
        return meanAcc, stdAcc
    end
         
    # Code adapted from the cross-validation version of the trainClassANN function
    k = maximum(crossValidationIndices)
    testAccsK = zeros(k)
    
    # Train with k different splits
    for ki in 1:k
        
        # Create the desired model passing the corresponding hyperparameters
        if (modelType == :SVM)
            model = SVC(kernel=modelHyperparameters["kernel"], degree=modelHyperparameters["degree"], 
                gamma=modelHyperparameters["gamma"], C=modelHyperparameters["C"])
        elseif (modelType == :DecisionTree)
            model = DecisionTreeClassifier(max_depth=modelHyperparameters["depth"], random_state=1)
        else
            model = KNeighborsClassifier(modelHyperparameters["numNeighbours"])
        end

        # Use the patterns with no k index for train
        trainingInputs = inputs[crossValidationIndices .!= ki, :]
        trainingTargets = targets[crossValidationIndices .!= ki]

        # Use the patterns with the k index for test
        testInputs = inputs[crossValidationIndices .== ki, :]
        testTargets = targets[crossValidationIndices .== ki]

        # Train the model
        model = fit!(model, trainingInputs, trainingTargets)

        # Compute the accuracy with the confusion matrix
        outputs = predict(model, testInputs)
        testAccsK[ki], _, _, _, _, _, _, _ = confusionMatrix(outputs, testTargets)
    end
    
    # Return the average and std of the metrics in the different k folds
    return mean(testAccsK), std(testAccsK)        
end;

function trainModelFullDataset(modelType::Symbol,
        modelHyperparameters::Dict{String, Any},
        trainingDataset::Tuple{AbstractArray{<:Real,2}, AbstractArray{<:Any,1}})

    
    @assert (in(modelType, [:ANN, :SVM, :kNN, :DecisionTree])) "Model type $(modelType) is not supported"
    
    inputs, targets = trainingDataset
                
    # Train an ANN model
    if (modelType == :ANN)
        targets = oneHotEncoding(targets)
        
        # we use the default transfer functions (sigmoid)
        model, _, _, _, _, _, _ = trainClassANN(modelHyperparameters["topology"], (inputs, targets);
            testDataset=(inputs, targets), maxEpochs=modelHyperparameters["maxEpochs"], 
            minLoss=modelHyperparameters["minLoss"], learningRate=modelHyperparameters["learningRate"],
            maxEpochsVal=modelHyperparameters["maxEpochsVal"])
        outputs = copy(model(inputs')')  # copy to convert the Adjoint to a Matrix type
        
    # Train an Scikit-Learn model
    else
        # Create the desired model passing the corresponding hyperparameters
         if (modelType == :SVM)
            model = SVC(kernel=modelHyperparameters["kernel"], degree=modelHyperparameters["degree"], 
                gamma=modelHyperparameters["gamma"], C=modelHyperparameters["C"])
        elseif (modelType == :DecisionTree)
            model = DecisionTreeClassifier(max_depth=modelHyperparameters["depth"], random_state=1)
        else
            model = KNeighborsClassifier(modelHyperparameters["numNeighbours"])
        end

        # Train the model
        model = fit!(model, inputs, targets)
        outputs = predict(model, inputs)
    end
    
    # Print the confusion matrix and return the model as well
    printConfusionMatrix(outputs, targets)
    return model, confusionMatrix(outputs, targets)      
end;

function trainClassEnsemble(estimators::AbstractArray{Symbol,1}, 
        modelsHyperparameters::AbstractArray{<:Dict, 1},     
        trainingDataset::Tuple{AbstractArray{<:Real,2}, AbstractArray{<:Any,1}},    
        kFoldIndices::Array{Int64,1})
    
    inputs, targets = trainingDataset
    
    # cross-validation variables
    k = maximum(kFoldIndices)
    testAccsK = zeros(k)
    
    # Train with k different splits
    for ki in 1:k
        
        # Use the patterns with no k index for train
        trainingInputs = inputs[kFoldIndices .!= ki, :]
        trainingTargets = targets[kFoldIndices .!= ki]
        
        # Use the patterns with the k index for test
        testInputs = inputs[kFoldIndices .== ki, :]
        testTargets = targets[kFoldIndices .== ki]
        
        # Create the models 
        models = []
        modelsNames = Dict(:SVM => "SVM", :DecisionTree => "DecisionTree", :kNN => "kNN", :ANN => "ANN", :MLP => "ANN")
        for (i, estimator) in enumerate(estimators)
            if (estimator == :SVM)
                model = SVC(
                    kernel=modelsHyperparameters[i]["kernel"], 
                    degree=modelsHyperparameters[i]["degree"], 
                    gamma=modelsHyperparameters[i]["gamma"], 
                    C=modelsHyperparameters[i]["C"]
                )
            elseif (estimator == :DecisionTree)
                model = DecisionTreeClassifier(max_depth=modelsHyperparameters[i]["depth"], random_state=1)
            elseif (estimator == :kNN)
                model = KNeighborsClassifier(modelsHyperparameters[i]["numNeighbours"])
            elseif (estimator == :ANN) || (estimator == :MLP)
                model = MLPClassifier(
                    hidden_layer_sizes=modelsHyperparameters[i]["topology"], 
                    max_iter=modelsHyperparameters[i]["maxEpochs"], 
                    #minLoss=modelHyperparameters["minLoss"],  there is no analog parameter for the minLoss
                    learning_rate_init=modelsHyperparameters[i]["learningRate"], 
                    early_stopping=modelsHyperparameters[i]["validationRatio"] > 0.0,
                    validation_fraction=modelsHyperparameters[i]["validationRatio"], 
                    n_iter_no_change=modelsHyperparameters[i]["maxEpochsVal"]
                )
            else
                error("Model type $(estimator) is not supported")
            end
            name = "$(modelsNames[estimator]) ($i)"
            push!(models, (name, deepcopy(model)))
        end
        
        # Create the ensemble and train
        model = VotingClassifier(estimators=models, n_jobs=-1)        
        fit!(model, trainingInputs, trainingTargets)

        # Compute the accuracy of the k-fold and save it
        testAccsK[ki] = score(model, testInputs, testTargets)
    end

    # Return the average and std of the metrics in the different k folds
    return mean(testAccsK), std(testAccsK)
end;

function trainClassEnsemble(estimators::AbstractArray{Symbol,1}, 
        modelsHyperparameters::Dict,     
        trainingDataset::Tuple{AbstractArray{<:Real,2}, AbstractArray{<:Any,1}},   
        kFoldIndices::Array{Int64,1})

    arrayHyperparameters = convert(Array{Dict{String,Any},1}, [])
    
    for (i, estimator) in enumerate(estimators)
        parameters = Dict()
        if (estimator == :SVM)
            parameters["kernel"] = modelsHyperparameters["kernel"]
            parameters["degree"] = modelsHyperparameters["degree"]
            parameters["gamma"] = modelsHyperparameters["gamma"]
            parameters["C"] = modelsHyperparameters["C"]
        elseif (estimator == :DecisionTree)
            parameters["depth"] = modelsHyperparameters["depth"]
        elseif (estimator == :kNN)
            parameters["numNeighbours"] = modelsHyperparameters["numNeighbours"]
        elseif (estimator == :ANN) || (estimator == :MLP)
            parameters["topology"] = modelsHyperparameters["topology"]
            parameters["maxEpochs"] = modelsHyperparameters["maxEpochs"]
            parameters["learningRate"] = modelsHyperparameters["learningRate"]
            parameters["validationRatio"] = modelsHyperparameters["validationRatio"]
            parameters["maxEpochsVal"] = modelsHyperparameters["maxEpochsVal"]
        end
        push!(arrayHyperparameters, deepcopy(parameters))
    end
    
    return trainClassEnsemble(estimators, arrayHyperparameters, trainingDataset, kFoldIndices)
end;

function trainClassEnsemble(baseEstimator::Symbol, 
        modelsHyperparameters::Dict,
        trainingDataset::Tuple{AbstractArray{<:Real,2}, AbstractArray{<:Any,1}},
        kFoldIndices::Array{Int64,1},
        numEstimators::Int=100)
    
    estimators = [baseEstimator for _ in 1:numEstimators]
    arrayHyperparameters = [modelsHyperparameters for _ in 1:numEstimators]
    return trainClassEnsemble(estimators, arrayHyperparameters, trainingDataset, kFoldIndices)
end;