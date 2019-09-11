import { IFCLayer, IFCLayerConfig, EActFunction } from "./types";
import { deepClone, isMatrix } from "./helper";
import { applyToVector, isActivationFunction } from "./activation-functions";

/* --------------------------------- Public --------------------------------- */

export function init(numOfInputs: number, outputLayer: IFCLayerConfig, hiddenLayers?: IFCLayerConfig[],
    noBias?: boolean): IFCLayer[]
{
    // prepare aux array, which holds th number of neurons for each layer except
    // the output layer
    let neuronsPerLayer: number[] = [numOfInputs]
    if (hiddenLayers)
        hiddenLayers.forEach(layer => neuronsPerLayer.push(layer.numOfNeurons) )

    // create hidden layers. Default activation function: Sigmoid
    let result: IFCLayer[] = []
    if (hiddenLayers) {
        result = hiddenLayers.map((layer, i) => {
            let neurons = neuronsPerLayer[i+1]
            // if bias is used, one additional weight per neuron is needed
            let weights = neuronsPerLayer[i] + (!noBias ? 1:  0)
            return <IFCLayer>{
                actFunction: layer.actFunction,
                weights: createMatrixWithRandomValues(neurons, weights)
            }
        })
    }

    // add output layer, Default activation function: ReLU
    // if bias is used, one additional weight per neuron is needed
    let lastLayerWeights = neuronsPerLayer[neuronsPerLayer.length - 1]
        + (!noBias ? 1:  0)
    result.push(<IFCLayer>{
        actFunction: outputLayer.actFunction,
        weights: createMatrixWithRandomValues(outputLayer.numOfNeurons, lastLayerWeights)
    })

    return result
}

export function calc(input: number[], layers: IFCLayer[]): number[] {
    return layers.reduce((result, layer) => {
        return calcLayerResult(result, layer).activated
    }, input)
}

export function train(_input: number[], expectedOutput: number[], learnRate: number,
    layers: IFCLayer[]): IFCLayer[]
{
    let input = deepClone(_input)

    // make reverse version of layers -> easier to handle backward propagation
    let layersReverse: IFCLayer[] = deepClone(layers)
    layersReverse.reverse()

    // FORWARD PROPAGATION: 

    // calc LayerResults and store all intermediate results
    let layerResults: ILayerResult[] = []
    let initLayerResult: ILayerResult = { activated: input, weighted: [] }
    layers.reduce((prevLayerResult, layer) => {
        let layerResult = calcLayerResult(prevLayerResult.activated, layer)
        layerResults.push(layerResult)
        return layerResult
    }, initLayerResult)
    layerResults.unshift(initLayerResult)

    // add bias neuron, if necessary. This is added on activation results only.
    layers.forEach((layer, i) => {
        if (layer.weights[0].length - layerResults[i].activated.length === 1)
            layerResults[i].activated.push(1)
    })

    // reverse layerResults -> prepare for backpropagation
    layerResults.reverse()

    // BACKPROPAGATION: 
    
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
    let deltas: number[][] = []
    layersReverse.forEach((_, i) => {
        deltas[i] = (i === 0)
            ? initialDelta(expectedOutput, layerResults[i], layersReverse[i].actFunction)
           :  updateDelta(deltas[i-1], layersReverse[i-1].weights, layerResults[i], 
                layersReverse[i].actFunction)
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

/* ------------------------------- Validation ------------------------------- */

export function isLayer(layer: IFCLayer): layer is IFCLayer {
    return 'actFunction' in layer && isActivationFunction(layer.actFunction)
        && 'weights' in layer  && isMatrix(layer.weights)
}

export function doAllLayersInAndOutputMatch(layers: IFCLayer[]): boolean {
    let inOutputs = layers.map(layer => getNumOfInAndOutputs(layer))
    for (let i = 1, ie = layers.length; i < ie; i++) {
        if (   inOutputs[i-1].outputs     !== inOutputs[i].inputs 
            && inOutputs[i-1].outputs + 1 !== inOutputs[i].inputs)
                return false
    }
    return true
}

export function doInAndOutputMatchWithNet(layers: IFCLayer[], numOfInputs: number, 
    numOfOutputs?: number): boolean
{
    let netInputs  = getNumOfInAndOutputs(layers[0]).inputs
    let netOutputs = getNumOfInAndOutputs(layers[layers.length - 1]).outputs
    return (numOfInputs === netInputs || numOfInputs + 1 === netInputs)
        && (!numOfOutputs || netOutputs === numOfOutputs)
}

function getNumOfInAndOutputs(layer: IFCLayer): {inputs:number, outputs:number} {
    return {
        inputs: layer.weights[0].length,
        outputs: layer.weights.length
    }
}

/* --------------------------------- Intern --------------------------------- */

interface ILayerResult {
    weighted: number[]  // weighted prev layers' results
    activated: number[] // activation function applied to results from weighted
}

function calcLayerResult(input: number[], layer: IFCLayer): ILayerResult {
    let result: ILayerResult = { weighted: [], activated: [] }

    result.weighted  = multiplyMatrixWithVector(layer.weights, input)
    result.activated = applyToVector(result.weighted, layer.actFunction)

    return result
}

function initialDelta(expectedOutput: number[], layerResult: ILayerResult, 
    actFunc: EActFunction): number[] 
{
    let gradients = applyToVector(layerResult.weighted, actFunc, true)

    return expectedOutput.map((_, i) => {
        return gradients[i] * (layerResult.activated[i] - expectedOutput[i])
    })
}

function updateDelta(prevDelta: number[], weightsToPrevLayer: number[][],
    layerResult: ILayerResult, actFunc: EActFunction): number[] 
{
    let gradients = applyToVector(layerResult.weighted, actFunc, true)

    // If there is a bias neuron, errorForThisLayer has a delta for that bias
    // as its last entry. This is not needed, so it will always be ignored.
    // Nonetheless, its calculated here because of laziness. ;-)
    let matrix = transposeMatrix(weightsToPrevLayer)
    let errorForThisLayer = multiplyMatrixWithVector(matrix, prevDelta)

    // Ignores bias neuron if there is one.
    return layerResult.weighted.map((_, i) => {
        return gradients[i] * errorForThisLayer[i]
    })
}

function updateWeights(delta: number[], weights: number[][], actResultsPrevLayer: number[],
    learnRate: number): number[][] 
{
    return weights.map((neuron, i) => {
        return neuron.map((weight, j) => {
            return weight - delta[i] * actResultsPrevLayer[j] * learnRate
        })
    })
}

/* --------------------------------- Helper --------------------------------- */

function createMatrixWithRandomValues(rows: number, columns: number): number[][] {
    let result: number[][] = []
    for (var i = 0; i < rows; i++) {
        let row: number[] = []
        for (var j = 0; j < columns; j++) {
            row.push(Math.random() + 0.01)
        }
        result.push(row)
    }
    return result
}

function multiplyMatrixWithVector(matrix: number[][], _vector: number[]): number[] {
    let vector = [..._vector]
    
    // add bias neuron if necessary
    if (matrix[0].length - _vector.length === 1)
        vector.push(1)

    return matrix.map(row => {
        return vector.reduce((result, _, i) => {
            return result + vector[i] * row[i]
        }, 0)
    })
}

function transposeMatrix(matrix: number[][]): number[][] {
    return matrix[0].map((_, i) => matrix.map(row => row[i]))
}