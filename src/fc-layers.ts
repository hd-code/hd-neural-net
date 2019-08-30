import { EActFunction, applyToVector } from "./activation-functions";
import { deepClone } from "./helper";

/* --------------------------------- Public --------------------------------- */

export interface ILayer {
    actFunction: EActFunction
    weights: number[][] // [neurons on this layer][weights for outputs from prev layer]
}

export interface ILayerConfig {
    actFunction?: EActFunction
    numOfNeurons: number
}

export function init(numOfInputs :number, outputLayer: ILayerConfig, hiddenLayers? :ILayerConfig[],
    noBias? :boolean) :ILayer[]
{
    // prepare aux array, which holds th number of neurons for each layer except
    // the output layer
    let neuronsPerLayer :number[] = [numOfInputs]
    if (hiddenLayers)
        hiddenLayers.forEach(layer => neuronsPerLayer.push(layer.numOfNeurons) )

    // add bias neuron if necessary
    // if (!noBias)
    //     neuronsPerLayer = neuronsPerLayer.map(number => number+1)

    // create hidden layers. Default activation function: Sigmoid
    let result :ILayer[] = []
    if (hiddenLayers) {
        result = hiddenLayers.map((layer, i) => {
            let neurons = neuronsPerLayer[i+1]
            let weights = neuronsPerLayer[i] + (!noBias ? 1 : 0)
            return <ILayer>{
                actFunction: layer.actFunction || EActFunction.SIGMOID,
                weights: createMatrixWithRandomValues(neurons, weights)
            }
        })
    }

    // add output layer, Default activation function: ReLU
    let lastLayer = neuronsPerLayer[neuronsPerLayer.length - 1] + (!noBias ? 1 : 0)
    result.push(<ILayer>{
        actFunction: outputLayer.actFunction || EActFunction.RELU,
        weights: createMatrixWithRandomValues(outputLayer.numOfNeurons, lastLayer)
    })

    return result
}

export function calc(input :number[], layers :ILayer[]) :number[] {
    return layers.reduce((result, layer) => {
        return calcLayerResult(result, layer).activated
    }, input)
}

export function train(input :number[], expectedOutput :number[], learnRate :number,
    layers :ILayer[]) :ILayer[]
{
    // make reverse version of layers -> easier to handle backward propagation
    let layersReverse :ILayer[] = deepClone(layers)
    layersReverse.reverse()

    // FORWARD PROPAGATION :

    // calc LayerResults and store all intermediate results
    let layerResults :ILayerResult[] = []
    let initLayerResult :ILayerResult = { activated: input, weighted: [] }
    layers.reduce((prevLayerResult, layer) => {
        let layerResult = calcLayerResult(prevLayerResult.activated, layer)
        layerResults.push(layerResult)
        return layerResult
    }, initLayerResult)
    layerResults.unshift(initLayerResult)

    // add bias neuron, if necessary. This is added on activation results.
    layers.forEach((layer, i) => {
        let diff = layer.weights[0].length - layerResults[i].activated.length
        while (0 < diff--) layerResults[i].activated.push(1)
    })

    // reverse layerResults -> prepare for backpropagation
    layerResults.reverse()

    // BACKPROPAGATION :
    
    // calc deltas
    // Delta basically tells how much the result is "off".
    // Calculation: f'(w) * e
    //      f' ... derivative of activation function
    //      w  ... weighted result, so the result before activation function was
    //              applied, basically layersResult.weighted[i]
    //      e  ... the error for this specific neuron. On the output layer this
    //              can be calculated using the expected output (the difference
    //              between calculated and expected output). On the hidden
    //              layers the error from the previous layers (ergo their delta)
    //              must be distributed to these layers neurons using the 
    //              corresponding weights. This is done by multiplying the prev
    //              delta with the transposed weight matrix.
    let deltas :number[][] = []
    layersReverse.forEach((layer, i) => {
        deltas[i] = (i === 0)
            ? initialDelta(expectedOutput, layerResults[i], layer)
            : updateDelta(deltas[i-1], layerResults[i], layer)
    })

    // update weights
    // Calculation: w - d * r * l
    //      w ... weight to be updated
    //      d ... delta for the corresponding neuron
    //      r ... layer result from previous layer with activation function
    //              applied. Corresponds to the weight for prev layers' neuron
    //      l ... learning rate. used to damp and smooth the learning process
    let result = layersReverse.map((layer, i) => {
        // layerResults has to be from i+1, because its the actual previous 
        // layers result
        layer.weights = updateWeights(deltas[i], layer.weights, layerResults[i+1].activated, learnRate)
        return layer
    })
    result.reverse() // reverse again, to restore initial order

    return result
}

/* --------------------------------- Intern --------------------------------- */

interface ILayerResult {
    weighted: number[]  // weighted prev layers' results
    activated: number[] // activation function applied to results from weighted
}

function calcLayerResult(input :number[], layer :ILayer) :ILayerResult {
    let result :ILayerResult = { weighted: [], activated: [] }

    result.weighted  = multiplyVectorWithMatrix(input, layer.weights)
    result.activated = applyToVector(result.weighted, layer.actFunction)

    return result
}

function initialDelta(expectedOutput :number[], outputLayerResult :ILayerResult, outputLayer :ILayer) :number[] {
    let gradients = applyToVector(outputLayerResult.weighted, outputLayer.actFunction, true)

    return expectedOutput.map((_, i) => {
        return gradients[i] * (outputLayerResult.activated[i] - expectedOutput[i])
    })
}

function updateDelta(prevDelta :number[], layerResult :ILayerResult, layer :ILayer) :number[] {
    let gradients = applyToVector(layerResult.weighted, layer.actFunction, true)

    // If there is a bias neuron, errorForThisLayer has a delta for that bias
    // as its last entry. This is not needed, so it will always be ignored.
    // Nonetheless, its calculated here because of laziness. ;-)
    let matrix = transposeMatrix(layer.weights)
    let errorForThisLayer = multiplyVectorWithMatrix(prevDelta, matrix)

    // Ignores bias neuron if there is one.
    return layerResult.weighted.map((_, i) => {
        return gradients[i] * errorForThisLayer[i]
    })
}

function updateWeights(delta :number[], weights :number[][], actResultsPrevLayer :number[],
    learnRate :number) :number[][] 
{
    return weights.map((neuron, i) => {
        return neuron.map((weight, j) => {
            return weight - delta[i] * actResultsPrevLayer[j] * learnRate
        })
    })
}

/* --------------------------------- Helper --------------------------------- */

function createMatrixWithRandomValues(rows :number, columns :number) :number[][] {
    let result :number[][] = []
    for (var i = 0; i < rows; i++) {
        let row :number[] = []
        for (var j = 0; j < columns; j++) {
            row.push(Math.random())
        }
        result.push(row)
    }
    return result
}

function multiplyVectorWithMatrix(_vector :number[], matrix :number[][]) :number[] {
    let vector = [..._vector]
    
    // add bias neuron if necessary
    let diff = matrix[0].length - _vector.length
    while (0 < diff--) vector.push(1)

    return matrix.map(row => {
        return vector.reduce((result, _, i) => {
            return result + vector[i] * row[i]
        }, 0)
    })
}

function transposeMatrix(matrix :number[][]) :number[][] {
    return matrix[0].map((_, i) => matrix.map(row => row[i]))
}