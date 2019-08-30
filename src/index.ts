import * as fcLayer from "./fc-layers";

interface IHDNeuralNet {
    title?: string
    description?: string
    createdAt: Date
    updatedAt: Date
    layers: fcLayer.ILayer[]
    learningRate: number
    precision: number
}

const DEFAULT_LEARNING_RATE = .01
const DEFAULT_PRECISION = .01

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

function checkNet(net : any) :net is IHDNeuralNet {
    if ('layers' ! in net || fcLayer.isFCNet(net.layers))
        return false

    net.learningRate = net.learningRate || DEFAULT_LEARNING_RATE
    net.precision = net.precision || DEFAULT_PRECISION
    return true
}
*/