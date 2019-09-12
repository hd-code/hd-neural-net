// import { INet, IFCLayer, IFCLayerConfig } from "./types";
import * as helper from "./helper";
import { EActFunction, applyToVector, isActivationFunction } from "./activation-functions";

/* --------------------------------- Public --------------------------------- */

export function init(numOfInputs :number, outputLayer :IFCLayerConfig, 
    hiddenLayers ?:IFCLayerConfig[], noBias ?:boolean) :INet
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
                actFunction: layer.actFunction || EActFunction.SIGMOID,
                weights: helper.createMatrixWithRandomValues(weights, neurons)
            }
        })
    }

    // add output layer, Default activation function: ReLU
    // if bias is used, one additional weight per neuron is needed
    let lastLayerWeights = neuronsPerLayer[neuronsPerLayer.length - 1]
        + (!noBias ? 1:  0)
    result.push(<IFCLayer>{
        actFunction: outputLayer.actFunction || EActFunction.RELU,
        weights: helper.createMatrixWithRandomValues(lastLayerWeights, outputLayer.numOfNeurons)
    })

    return result
}

export function calc(input :number[], layers :INet) :number[] {
    return layers.reduce((result, layer) => {
        return calcLayerResult(result, layer).activated
    }, input)
}

export function train(_input :number[], expectedOutput :number[], learnRate :number,
    layers :INet) :INet
{
    let input = helper.deepClone(_input)

    // make reverse version of layers -> easier to handle backward propagation
    let layersReverse: IFCLayer[] = helper.deepClone(layers)
    layersReverse.reverse()

    // FORWARD PROPAGATION: 

    // calc LayerResults and store all intermediate results
    let layerResults: ILayerResult[] = calcLayerResults(input, layers)
    layerResults.unshift(<ILayerResult>{ activated: input, weighted: [] })

    // add bias neuron, if necessary. This is added on activation results only.
    layers.forEach((layer, i) => {
        if (layer.weights.length - layerResults[i].activated.length === 1)
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

export function isValid(layers :INet, input ?:number[], output ?:number[]) :boolean {
    if (!helper.isArrayOf(layers, isLayer)) {
        console.error('Invalid data type for the layers')
        return false
    }
    if (!doAllLayersInAndOutputMatch(layers)) {
        console.error('Some layers produce the wrong number of outputs for the following layer')
        return false
    }
    if (input && input.length !== getNumOfInAndOutputs(layers[0]).inputs) {
        console.error('Number of inputs don\'t match with the neural net')
        return false
    }
    if (output && output.length !== getNumOfInAndOutputs(layers[layers.length-1]).outputs) {
        console.error('Number of outputs don\'t match with the neural net')
        return false
    }
    return true
}

/* --------------------------------- Types ---------------------------------- */

interface INet extends Array<IFCLayer> {}

interface IFCLayer {
    actFunction: EActFunction
    weights: number[][] // [neurons on prev layer][neurons on this layer]
}

interface IFCLayerConfig {
    actFunction?: EActFunction
    numOfNeurons: number
}

/* ------------------------------- Validation ------------------------------- */

function isLayer(layer: IFCLayer): layer is IFCLayer {
    return 'actFunction' in layer && isActivationFunction(layer.actFunction)
        && 'weights' in layer  && helper.isMatrix(layer.weights)
}

function doAllLayersInAndOutputMatch(layers: IFCLayer[]): boolean {
    let inOutputs = layers.map(layer => getNumOfInAndOutputs(layer))
    for (let i = 1, ie = layers.length; i < ie; i++) {
        if (   inOutputs[i-1].outputs     !== inOutputs[i].inputs // no bias
            && inOutputs[i-1].outputs + 1 !== inOutputs[i].inputs // with bias
        )
            return false
    }
    return true
}

function getNumOfInAndOutputs(layer: IFCLayer): {inputs:number, outputs:number} {
    return {
        inputs: layer.weights.length,
        outputs: layer.weights[0].length
    }
}

/* --------------------------------- Intern --------------------------------- */

interface ILayerResult {
    weighted: number[]  // weighted prev layers' results
    activated: number[] // activation function applied to results from weighted
}

function calcLayerResults(_input: number[], layers: IFCLayer[]): ILayerResult[] {
    let input = helper.deepClone(_input)

    let result :ILayerResult[] = []
    layers.forEach(layer => {
        result.push(calcLayerResult(input, layer))
        input = result[result.length - 1].activated
    })
    return result
}

function calcLayerResult(_input: number[], layer: IFCLayer): ILayerResult {
    // append bias neuron if necessary
    let input = helper.deepClone(_input)
    if (input.length + 1 === layer.weights.length)
        input.push(1)

    let result: ILayerResult = { weighted: [], activated: [] }

    result.weighted  = helper.multiplyVectorWithMatrix(input, layer.weights)
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
    let matrix = helper.transposeMatrix(weightsToPrevLayer)
    let errorForThisLayer = helper.multiplyVectorWithMatrix(prevDelta, matrix)

    // Ignores bias neuron if there is one.
    return layerResult.weighted.map((_, i) => {
        return gradients[i] * errorForThisLayer[i]
    })
}

function updateWeights(delta: number[], weights: number[][], actResultsPrevLayer: number[],
    learnRate: number): number[][] 
{
    return weights.map((_, i) => {
        return weights[i].map((_, j) => {
            return weights[i][j] - delta[j] * actResultsPrevLayer[i] * learnRate
        })
    })
}