import { IHDNeuralNet, INetOptions, ITrainingData, IFCLayerConfig, EActFunction } from "./types";
import { deepClone, isNumber, isVector, isMatrix, isArrayOf, isString, isBool } from "./helper";
import * as fcLayer from "./fc-layers";
import { isActivationFunction } from "./activation-functions";

/* --------------------------------- Public --------------------------------- */

/** Initializes a neural net */
export function init(numOfInputs: number, numOfOutputs: number, neuronsPerHiddenLayer?: number[],
    additional?: INetOptions): IHDNeuralNet|null
{
    if (  !isNumber(numOfInputs) || !isNumber(numOfOutputs)
        || neuronsPerHiddenLayer && !isVector(neuronsPerHiddenLayer)) 
    {
        console.error('no numbers provided')
        return null
    }

    let noBias = additional && isBool(additional.noBias) ? additional.noBias : undefined

    let actFunctions = setActivationFunctions(neuronsPerHiddenLayer, additional)

    let hiddenLayers = neuronsPerHiddenLayer && neuronsPerHiddenLayer.map((numOfNeurons, i) => {
        return <IFCLayerConfig>{
            actFunction: actFunctions[i],
            numOfNeurons: numOfNeurons
        }
    })

    let outputLayer: IFCLayerConfig = {
        actFunction: actFunctions[actFunctions.length - 1],
        numOfNeurons: numOfOutputs
    }

    let net: IHDNeuralNet = {
        createdAt: new Date,
        updatedAt: new Date,
        learningRate: additional && isNumber(additional.learningRate) 
                ? additional.learningRate 
                : DEFAULT_LEARNING_RATE,
        precision: additional && isNumber(additional.precision) 
                ? additional.precision 
                : DEFAULT_PRECISION,
        layers: fcLayer.init(numOfInputs, outputLayer, hiddenLayers, noBias)
    }

    if (additional && isString(additional.title))
        net.title = additional.title
    if (additional && isString(additional.description))
        net.description = additional.description

    return net
}

export function calc(_input: number[]|number[][], net: IHDNeuralNet): number[]|number[][]|null {
    if (!isHDNeuralNet(net)) {
        console.error('Provided neural net is invalid')
        return null
    }

    let singleInput = false
    let input :number[][] = []
    if (isVector(_input))      input = [_input], singleInput = true
    else if (isMatrix(_input)) input = _input
    else {
        console.error('Input has to be an array of numbers or a set of such')
        return null
    }

    const allValid = input.reduce((result,inputSet, i) => {
        if (!fcLayer.doInAndOutputMatchWithNet(net.layers, inputSet.length)) {
            let error = singleInput
                ? 'Wrong number of input values for this net'
                : 'Wrong number of input values on the ' + (i+1) + '. input set'
            console.error(error)
            return false
        }
        return result
    }, true)
    if (!allValid)
        return null

    const result = input.map(input => fcLayer.calc(input, net.layers))
    return singleInput ? result[0] : result
}

export function train(_data: ITrainingData|ITrainingData[], _net: IHDNeuralNet,
    logging ?:number, learningRate?: number, precision?: number): IHDNeuralNet|null 
{
    if (!isHDNeuralNet(_net)) {
        console.error('Provided neural net is invalid')
        return null
    }

    let data :ITrainingData[] = []
    if (isTrainingData(_data))
        data = [_data]
    else if (isArrayOf(data, isTrainingData))
        data = _data
    else {
        console.error('Provided data is no valid training data set')
        return null
    }

    const allValid = data.reduce<boolean>((valid, d, i) => {
        if (!fcLayer.doInAndOutputMatchWithNet(_net.layers, d.input.length, d.output.length)) {
            console.error('Number of inputs or expected outputs don\'t match with the provided neural net for ' + (i+1) + '. training data set')
            return false
        }
        return valid
    }, true)
    if (!allValid)
        return null

    let net = deepClone(_net)

    const expOutputs = data.map(data => data.output)
    const prec      = isNumber(precision)    ? precision    : net.precision
    const learnRate = isNumber(learningRate) ? learningRate : net.learningRate

    let epoche = 0
    let layers = net.layers
    let output = data.map(d => fcLayer.calc(d.input, layers))
    let highestDifference = getHighestDifference(output, expOutputs)

    while (highestDifference > prec) {
        epoche++

        layers = data.reduce((layers, d) => {
            return fcLayer.train(d.input, d.output, learnRate, layers)
        }, layers)
        output = data.map(d => fcLayer.calc(d.input, layers))

        highestDifference = getHighestDifference(output, expOutputs)
        if (logging && epoche % logging === 0)
            console.log('Highest Net Error:', highestDifference)
    }
    console.log('Training finished after:', epoche, 'iterations')

    net.layers = layers
    net.updatedAt = new Date
    return net
}

/* --------------------------------- Intern --------------------------------- */

const DEFAULT_LEARNING_RATE = .01
const DEFAULT_PRECISION = .01

const DEFAULT_ACT_FUNCTION_HIDDEN_LAYERS = EActFunction.SIGMOID
const DEFAULT_ACT_FUNCTION_OUTPUT_LAYER  = EActFunction.RELU

function isHDNeuralNet(net :IHDNeuralNet) :net is IHDNeuralNet {
    if (!('layers' in net)) {
        console.error('This net doesn\'t contain any layers')
        return false
    }

    if (!isArrayOf(net.layers, fcLayer.isLayer)) {
        console.error('The nets layers are invalid')
        return false
    }

    if (!fcLayer.doAllLayersInAndOutputMatch(net.layers)) {
        console.error('Some of the layers outputs don\'t match the following layers inputs')
        return false
    }

    // correct learning rate and precision if that's necessary
    net.learningRate = isNumber(net.learningRate) ? net.learningRate:  DEFAULT_LEARNING_RATE
    net.precision = isNumber(net.precision) ? net.precision:  DEFAULT_PRECISION

    return true
}

function isTrainingData(data: any): data is ITrainingData {
    return 'input'  in data && isVector(data.input)
        && 'output' in data && isVector(data.output)
}

function setActivationFunctions(neuronsPerHiddenLayer?: number[], o?: INetOptions):
    EActFunction[] 
{
    let result: EActFunction[] = []

    let allLayers :EActFunction|undefined
    let allHiddenLayers :EActFunction|undefined
    let hiddenLayers :EActFunction[]
    let outputLayer :EActFunction|undefined

    if (o && o.activationFunctions) {
        let t = o.activationFunctions

        if (t.allLayers !== undefined && isActivationFunction(t.allLayers))
            allLayers = t.allLayers

        if (t.allHiddenLayers !== undefined && isActivationFunction(t.allHiddenLayers))
            allHiddenLayers = t.allHiddenLayers

        if (t.hiddenLayers !== undefined && isArrayOf(t.hiddenLayers, isActivationFunction))
            hiddenLayers = t.hiddenLayers

        if (t.outputLayer !== undefined && isActivationFunction(t.outputLayer))
            outputLayer = t.outputLayer
    }

    neuronsPerHiddenLayer && neuronsPerHiddenLayer.forEach((_,i) => {
        let a = hiddenLayers && hiddenLayers[i] ? hiddenLayers[i]
              : allHiddenLayers !== undefined ? allHiddenLayers
              : allLayers       !== undefined ? allLayers
              : DEFAULT_ACT_FUNCTION_HIDDEN_LAYERS
        
        result.push(a)
    })

    let a = outputLayer !== undefined ? outputLayer
          : allLayers   !== undefined ? allLayers 
          : DEFAULT_ACT_FUNCTION_OUTPUT_LAYER
    result.push(a)

    return result
}

function getHighestDifference(x :number[][], y :number[][]) :number {
    return x.reduce((diff, _, i) => {
        return x[i].reduce((diff, _, j) => {
            return Math.max(Math.abs(x[i][j] - y[i][j]), diff)
        }, diff)
    }, 0)
}