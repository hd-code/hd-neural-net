import { deepClone, isNumber, isVector, isMatrix, isArrayOf } from "./helper";
import * as fcLayer from "./fc-layers";
import * as fcValid from "./fc-validation";
import { EActFunction, isActivationFunction } from "./activation-functions";

/* --------------------------------- Public --------------------------------- */

// TODO: INput Output check auslagern


export interface IHDNeuralNet {
    title?: string
    description?: string
    createdAt: Date
    updatedAt: Date
    learningRate: number
    precision: number
    layers: fcLayer.ILayer[]
}

export interface IOptions {
    noBias?: boolean
    learningRate?: number
    precision?: number

    title?: string
    description?: string

    activationFunctions?: {
        hiddenLayers?: EActFunction[]
        outputLayer?: EActFunction
    }
}

export function init(numOfInputs: number, numOfOutputs: number, neuronsPerHiddenLayer?: number[],
    additional?: IOptions): IHDNeuralNet|null
{
    if (!isNumber(numOfInputs) || !isNumber(numOfOutputs
        || neuronsPerHiddenLayer && !isVector(neuronsPerHiddenLayer))) 
    {
        console.error('no numbers provided')
        return null
    }

    let actFunctions = setActivationFunctions(neuronsPerHiddenLayer, additional)

    let hiddenLayers = neuronsPerHiddenLayer && neuronsPerHiddenLayer.map((numOfNeurons, i) => {
        return <fcLayer.ILayerConfig>{
            actFunction: actFunctions[i],
            numOfNeurons: numOfNeurons
        }
    })

    let outputLayer: fcLayer.ILayerConfig = {
        actFunction: actFunctions[actFunctions.length - 1],
        numOfNeurons: numOfOutputs
    }

    let net: IHDNeuralNet = {
        createdAt: new Date,
        updatedAt: new Date,
        learningRate: additional && additional.precision || DEFAULT_PRECISION,
        precision: additional && additional.precision || DEFAULT_PRECISION,
        layers: fcLayer.init(numOfInputs, outputLayer, hiddenLayers, additional && additional.noBias)
    }

    if (additional && additional.title)       net.title       = additional.title
    if (additional && additional.description) net.description = additional.description

    return net
}

export function calc(input: number[]|number[][], net: IHDNeuralNet): number[]|number[][]|null {
    if (!isHDNeuralNet(net)) {
        console.error('Provided neural net is invalid')
        return null
    }

    if (isMatrix(input)) {
        return calcSet(input, net)
    }

    if (isVector(input)) {
        return calcSingle(input, net)
    }

    console.error('Input has to be an array of numbers or a set of such')
    return null
}

export interface ITraingData {
    input: number[]
    output: number[]
}

export function train(data: ITraingData|ITraingData[], net: IHDNeuralNet,
    learningRate?: number, precision?: number): IHDNeuralNet|null 
{
    if (!isHDNeuralNet(net)) {
        console.error('Provided neural net is invalid')
        return null
    }

    precision    = isNumber(precision)    ? precision    : net.precision
    learningRate = isNumber(learningRate) ? learningRate : net.learningRate

    if (isTrainingData(data)) {
        return trainSingle(data, net, learningRate, precision)
    }

    if (isArrayOf(data, isTrainingData)) {
        return trainSet(data, net, learningRate, precision)
    }

    console.error('Provided data is no valid training data set')
    return null
}

/* --------------------------------- Intern --------------------------------- */

const DEFAULT_LEARNING_RATE = .01
const DEFAULT_PRECISION = .01

const DEFAULT_ACT_FUNCTION_HIDDEN_LAYERS = EActFunction.SIGMOID
const DEFAULT_ACT_FUNCTION_OUTPUT_LAYER  = EActFunction.RELU

function isHDNeuralNet(net: IHDNeuralNet): net is IHDNeuralNet {
    if (!('layers' in net) || !fcValid.isFCNet(net.layers)) {
        return false
    }

    net.learningRate = isNumber(net.learningRate) ? net.learningRate:  DEFAULT_LEARNING_RATE
    net.precision = isNumber(net.precision) ? net.precision:  DEFAULT_PRECISION
    return true
}

function setActivationFunctions(neuronsPerHiddenlayer?: number[], o?: IOptions): EActFunction[] {
    let result: EActFunction[] = []

    neuronsPerHiddenlayer && neuronsPerHiddenlayer.forEach((_,i) => {
        let a = o && o.activationFunctions && o.activationFunctions.hiddenLayers
            && isActivationFunction(o.activationFunctions.hiddenLayers[i])
            ? o.activationFunctions.hiddenLayers[i]
           :  DEFAULT_ACT_FUNCTION_HIDDEN_LAYERS
        
        result.push(a)
    })

    let a = o && o.activationFunctions && o.activationFunctions.outputLayer
        && isActivationFunction(o.activationFunctions.outputLayer)
        ? o.activationFunctions.outputLayer
       :  DEFAULT_ACT_FUNCTION_OUTPUT_LAYER
    result.push(a)

    return result
}

function calcSingle(input: number[], net: IHDNeuralNet): number[]|null {
    // if (fcValid.getNumOfNetInAndOutputs(net.layers).inputs !== input.length) {
    //     console.error('Wrong number of input values for this net')
    //     return null
    // }

    return fcLayer.calc(input, net.layers)
}

function calcSet(inputs: number[][], net: IHDNeuralNet): number[][]|null {
    // check if number of inputs matches with net inputs
    let netInputs = fcValid.getNumOfNetInAndOutputs(net.layers).inputs
    let allValid = inputs.reduce((result,input, i) => {
        if (netInputs < input.length) {
            result = false
            console.error('Wrong number of input values on the ' + (i+1) + '. data set')
        }
        return result
    }, true)

    return allValid ? inputs.map(input => fcLayer.calc(input, net.layers)):  null
}

function isTrainingData(data: any): data is ITraingData {
    return 'input'  in data && isVector(data.input)
        && 'output' in data && isVector(data.output)
}

function trainSingle(data: ITraingData, _net: IHDNeuralNet, learningRate:number,
    precision:number): IHDNeuralNet|null
{
    let net = deepClone(_net)
    let layers = net.layers

    if (!doInAndOutputsMatch(data, layers)) 
    {
        console.error('Number of inputs or expected outputs don\'t match with the provided neural net')
        return null
    }

    let output = fcLayer.calc(data.input, layers)
    let epoche = 0

    while(!closeEnough(data.output, output, precision)) {
        layers = fcLayer.train(data.input, data.output, learningRate, layers)
        output = fcLayer.calc(data.input, layers)
        epoche++
    }
    console.log('Training finished after:', epoche, 'iterations')

    net.layers = layers
    net.updatedAt = new Date
    return net
}

function trainSet(data: ITraingData[], _net: IHDNeuralNet, learningRate:number,
    precision:number): IHDNeuralNet|null
{
    let net = deepClone(_net)
    let layers = net.layers

    if (!data.reduce<boolean>((valid, d, i) => {
        if (!doInAndOutputsMatch(d, layers)) {
            console.error('Number of inputs or expected outputs don\'t match with the provided neural net for ' + (i+1) + '. training data set')
            return false
        }
        return valid
    }, true))
        return null

    let output = data.map(d => fcLayer.calc(d.input, layers))
    let epoche = 0

    let expOutputs = data.map(d => d.output)
    while(!closeEnoughSet(expOutputs, output, precision)) {
        layers = data.reduce((layers, d) => {
            return fcLayer.train(d.input, d.output, learningRate, layers)
        }, layers)
        output = data.map(d => fcLayer.calc(d.input, layers))
        epoche++
        if (epoche % 1000 === 0) console.log(output)
    }
    console.log('Training finished after:', epoche, 'iterations')

    net.layers = layers
    net.updatedAt = new Date
    return net
}

function doInAndOutputsMatch(data :ITraingData, layers :fcLayer.ILayer[]) :boolean {
    let netInAndOutputs = fcValid.getNumOfNetInAndOutputs(layers)
    return netInAndOutputs.inputs  >= data.input.length
        && netInAndOutputs.outputs === data.output.length
}

function closeEnough(x: number[], y: number[], delta: number): boolean {
    for (let i = 0, ie = x.length; i < ie; i++) {
        if (Math.abs(x[i] - y[i]) > delta)
            return false
    }
    return true
}

function closeEnoughSet(x: number[][], y: number[][], delta: number): boolean {
    for (let i = 0, ie = x.length; i < ie; i++) {
        if (!closeEnough(x[i], y[i], delta))
            return false
    }
    return true
}