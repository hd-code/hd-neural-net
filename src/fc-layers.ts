import { createMatrixWithRandomValues, deepClone, multiplyVectorWithMatrix, transposeMatrix } from "./helper";
import { EActFunc, applyToVector } from "./activation-functions";

/* --------------------------------- Public --------------------------------- */

export interface IFCLayer {
    actFunc: EActFunc
    weights: number[][] // [prevNeuron][thisLayersNeuron]
}

export interface IFCLayerConfig {
    actFunc: EActFunc
    numOfNeurons: number
}

export function init(numOfInputs: number, layers: IFCLayerConfig[], noBias?: boolean): IFCLayer[] {
    const neuronsPerLayer = [numOfInputs, ...layers.map(layer => layer.numOfNeurons)]
    return layers.map((layerConf, i) => {
        return {
            actFunc: layerConf.actFunc,
            weights: createMatrixWithRandomValues(
                neuronsPerLayer[i] + (!noBias ? 1 : 0),
                neuronsPerLayer[i+1]
            )
        }
    })
}

export function calc(layers: IFCLayer[], values: number[]): number[] {
    const  layerResults = calcLayerResults(values, layers)
    return layerResults[layerResults.length - 1].activated
}

export function train(layers: IFCLayer[], _input: number[], expOutput: number[], learnRate: number): IFCLayer[] {
    let input = deepClone(_input)

    // make reverse version of layers -> easier to handle backward propagation
    let layersReverse: IFCLayer[] = deepClone(layers)
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
            ? initialDelta(expOutput, layerResults[i], layersReverse[i].actFunc)
           :  updateDelta(deltas[i-1], layersReverse[i-1].weights, layerResults[i], 
                layersReverse[i].actFunc)
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

function calcLayerResults(_input: number[], layers: IFCLayer[]): ILayerResult[] {
    let input = deepClone(_input)

    let result :ILayerResult[] = []
    layers.forEach(layer => {
        result.push(calcLayerResult(input, layer))
        input = result[result.length - 1].activated
    })
    return result
}

function calcLayerResult(_input: number[], layer: IFCLayer): ILayerResult {
    // append bias neuron if necessary
    let input = deepClone(_input)
    if (input.length + 1 === layer.weights.length)
        input.push(1)

    let result: ILayerResult = { weighted: [], activated: [] }

    result.weighted  = multiplyVectorWithMatrix(input, layer.weights)
    result.activated = applyToVector(result.weighted, layer.actFunc)

    return result
}

function initialDelta(expOutput: number[], layerResult: ILayerResult, 
    actFunc: EActFunc): number[] 
{
    let gradients = applyToVector(layerResult.weighted, actFunc, true)

    return expOutput.map((_, i) => {
        return gradients[i] * (layerResult.activated[i] - expOutput[i])
    })
}

function updateDelta(prevDelta: number[], weightsToPrevLayer: number[][],
    layerResult: ILayerResult, actFunc: EActFunc): number[] 
{
    let gradients = applyToVector(layerResult.weighted, actFunc, true)

    // If there is a bias neuron, errorForThisLayer has a delta for that bias
    // as its last entry. This is not needed, so it will always be ignored.
    // Nonetheless, its calculated here because of laziness. ;-)
    let matrix = transposeMatrix(weightsToPrevLayer)
    let errorForThisLayer = multiplyVectorWithMatrix(prevDelta, matrix)

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