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
    // FORWARD PROPAGATION
    const layerResults = calcLayerResults(_input, layers)

    // BACKWARD PROPAGATION

    const actRes = [_input, ...layerResults.map(lr => lr.activated)]
    const actResults = actRes.map(ar => [...ar, 1])
    const deltas = calcDelta(layers, layerResults, expOutput)

    return layers.map((_,i) => {
        const newWeights = updateWeights(deltas[i], layers[i].weights, actResults[i], learnRate)
        return <IFCLayer>{
            actFunc: layers[i].actFunc,
            weights: newWeights
        }
    })
}

/* --------------------------------- Intern --------------------------------- */

interface ILayerResult {
    weighted: number[]  // weighted prev layers' results
    activated: number[] // activation function applied to results from weighted
}

function calcLayerResults(_input: number[], layers: IFCLayer[]): ILayerResult[] {
    let input = deepClone(_input)
    let result: ILayerResult[] = []
    layers.forEach((layer, i) => {
        result.push( calcLayerResult(input, layer) )
        input = result[i].activated
    })
    return result
}

function calcLayerResult(_input: number[], layer: IFCLayer): ILayerResult {
    // append bias neuron, is ignored if not needed
    let input = deepClone(_input)
    input.push(1)

    let result: ILayerResult = { weighted: [], activated: [] }

    result.weighted  = multiplyVectorWithMatrix(input, layer.weights)
    result.activated = applyToVector(result.weighted, layer.actFunc)

    return result
}

function calcDelta(layers: IFCLayer[], layerResults: ILayerResult[], expOutput: number[]):number[][] {
    const gradients = calcGradients(layers, layerResults)

    // calc delta on output layer
    //      gradient * (actualResult - expectedOutput)
    const errorOnOutputLayer = layerResults[layerResults.length-1].activated.map((res, i) => {
        return gradients[gradients.length-1][i] * (res - expOutput[i])
    })

    // calc deltas on hidden layers
    //      gradient * (prevDelta * WeightMatrix^T)
    //          WeightMatrix^T...weight matrix of current layer, transposed to 
    //                           distribute the delta backwards
    return layers.reduceRight((result, _, i) => {
        if (i === 0)
            return result
        const prevDelta = result[0]
        const matrix = transposeMatrix(layers[i].weights)
        const errorOnCurrentLayer = multiplyVectorWithMatrix(prevDelta, matrix)
        const delta = gradients[i].map((_, j) => {
            return gradients[i][j] * errorOnCurrentLayer[j]
        })
        return [delta, ...result]
    }, [errorOnOutputLayer])
}

// calc gradients for each layer
//      gradient = f'(x)
//          f...activation function
//          x...weighted result of prevLayer (layerResults[i].weighted)
function calcGradients(layers: IFCLayer[], layerResults: ILayerResult[]): number[][] {
    const ActFuncs = layers.map(layer => layer.actFunc)
    const valBeforeAct = layerResults.map(res => res.weighted)
    return ActFuncs.map((_,i) => applyToVector(valBeforeAct[i], ActFuncs[i], true))
}

function updateWeights(delta: number[], weights: number[][], actResultsPrevLayer: number[], learnRate: number): number[][] {
    return weights.map((_, i) => {
        return weights[i].map((_, j) => {
            return weights[i][j] - delta[j] * actResultsPrevLayer[i] * learnRate
        })
    })
}