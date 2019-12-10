import { mulVecMat, subVec, transposeMatrix, mulVec, scaleVec, createRandomMatrix } from './math';
import { EActFunc, applyToVector } from './activation-functions';
import { deepClone } from './helper';

/* --------------------------------- Public --------------------------------- */

export interface IFCLayer {
    actFunc: EActFunc
    weights: number[][] // [prevNeuron][nextNeuron]
}

export interface IFCLayerConfig {
    actFunc: EActFunc
    numOfNeurons: number
}

export function init(numOfInputs: number, layers: IFCLayerConfig[], noBias?: boolean): IFCLayer[] {
    const neuronsPerLayer = [numOfInputs, ...layers.map(lc => lc.numOfNeurons)]
    return layers.map((_, i) => ({
        actFunc: layers[i].actFunc,
        weights: createRandomMatrix(
            neuronsPerLayer[i] + (noBias ? 0 : 1),
            neuronsPerLayer[i+1]
        )
    }))
}

export function calc(input: number[], layers: IFCLayer[]): number[] {
    const results = calcLayerOutputs(input, layers)
    return results[results.length-1]
}

export function train(input: number[], expOutput: number[], layers: IFCLayer[], learnRate: number): IFCLayer[] {
    const layerOutputs = calcLayerOutputs(input, layers, true)
    const layerDerivatives = calcLayersDerivative(input, layers)

    const actOutput = layerOutputs[layerOutputs.length-1]
    const netError  = subVec(actOutput, expOutput)
    const numOfLastLayer = layers.length - 1

    /*
        delta tells us how much our calculation is off.

        delta on output layer:
            delta = f'(x) * (calcOutput - expOutput)
        delta on hidden layers:
            delta = f'(x) * deltaNextLayer * weights^T

    */
    const deltas = layers.reduceRight((result, _, i) => {
        // delta rule for output layer
        if (i === numOfLastLayer) {
            const delta = mulVec(layerDerivatives[i], netError)
            return [ delta ]
        }

        // delta rule on hidden layers
        const weightedDeltaFromPrev = mulVecMat(result[0], transposeMatrix(layers[i+1].weights))
        const delta = mulVec(layerDerivatives[i], weightedDeltaFromPrev)

        return [delta, ...result]
    }, <number[][]>[])

    // update weights and return new layers
    return layers.map((layer, layerI) => ({
        actFunc: layer.actFunc,
        weights: layer.weights.map((row, i) => {
            const rowInput = layerOutputs[layerI][i]
            return subVec(row, scaleVec(learnRate * rowInput, deltas[layerI]))
        })
    }))
}

/* --------------------------------- Intern --------------------------------- */

/** Calculates the values for each neuron on each layer. So the first entry in this array are the input values, followed by the results on all hidden layers. The last entry contains the result on the output layer and is therefore the final output of this net. */
function calcLayerOutputs(input: number[], layers: IFCLayer[], withBias?:boolean): number[][] {
    const outputLayer = layers.length

    const result = layers.reduce((result, layer) => {
        return [...result, calcLayerOutput(result[result.length-1], layer)]
    }, [input])

    return withBias && hasBias(input, layers)
        ? result.map((layer, i) => i === outputLayer ? layer : [...layer, 1])
        : result
}

/** Calculates the derivative of the activation function on all hidden layers plus the output layer. */
function calcLayersDerivative(input: number[], layers: IFCLayer[]): number[][] {
    const layerResults = calcLayerOutputs(input, layers)
    return layers.map((layer, i) => calcLayerOutput(layerResults[i], layer, true))
}

/** Calculates either the output for a given layer or the derivative of the activation function on this layer. This is determined by the derivative flag */
function calcLayerOutput(_input: number[], layer: IFCLayer, derivative?: boolean): number[] {
    // add bias neuron
    let input = deepClone(_input)
    while (input.length < layer.weights.length) input.push(1)

    // calc result
    const result = mulVecMat(input, layer.weights)
    return applyToVector(result, layer.actFunc, derivative)
}

function hasBias(input: number[], layers: IFCLayer[]): boolean {
    return input.length + 1 === layers[0].weights.length
}