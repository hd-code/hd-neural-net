import { deepClone, isNumber, isVector, isMatrix } from "./helper";
import * as fcLayer from "./fc-layers";
import * as fcValid from "./fc-validation";

/* --------------------------------- Public --------------------------------- */

interface IHDNeuralNet {
    title?: string
    description?: string
    createdAt: Date
    updatedAt: Date
    learningRate: number
    precision: number
    layers: fcLayer.ILayer[]
}

function init() {}

function calc(input :number[], net :IHDNeuralNet) :number[]|null {
    let valid = isVector(input) && isHDNeuralNet(net) 
        && fcValid.getNumOfNetInAndOutputs(net.layers).inputs === input.length
    return valid ? fcLayer.calc(input, net.layers) : null
}

function calcSet(inputs :number[][], net :IHDNeuralNet) :number[][]|null {
    if (!isMatrix(inputs) || !isHDNeuralNet(net))
        return null

    let netInputs = fcValid.getNumOfNetInAndOutputs(net.layers).inputs
    let allValid = inputs.reduce((result,input, i) => {
        if (netInputs !== input.length) {
            result = false
            console.error('Wrong number of input values on the', i+1+'.','data set')
        }
        return result
    }, true)

    return allValid ? inputs.map(input => fcLayer.calc(input, net.layers)) : null
}

interface ITraingData {
    input: number[]
    output: number[]
}

function train(data :ITraingData, _net :IHDNeuralNet, learningRate?:number, precision?:number) :IHDNeuralNet {
    // TODO: Type Checking

    let net = deepClone(_net)
    let result = net.layers

    net.updatedAt = new Date
    return net
}

function trainSet(dataSet :ITraingData[], _net :IHDNeuralNet, learningRate?:number, precision?:number) :IHDNeuralNet {
    let net = deepClone(_net)
    let result = net.layers

    net.updatedAt = new Date
    return net
}

/* --------------------------------- Intern --------------------------------- */

const DEFAULT_LEARNING_RATE = .01
const DEFAULT_PRECISION = .01

function isHDNeuralNet(net :IHDNeuralNet) :net is IHDNeuralNet {
    if ('layers' ! in net || fcValid.isFCNet(net.layers)) {
        return false
    }

    net.learningRate = isNumber(net.learningRate) ? net.learningRate : DEFAULT_LEARNING_RATE
    net.precision = isNumber(net.precision) ? net.precision : DEFAULT_PRECISION
    return true
}

/*
export function init() :IHDNeuralNet {
    let result :IHDNeuralNet = {
        createdAt: new Date,
        updatedAt: new Date,
        learningRate: DEFAULT_LEARNING_RATE,
        precision: DEFAULT_PRECISION,
        layers: fcLayer.init()
    }
    return result
}
*/