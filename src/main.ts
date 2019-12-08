import { EActFunc } from "./activation-functions";
import * as fcLayers from "./fc-layers";
import { deepClone } from "./helper";

/* --------------------------------- Types ---------------------------------- */

export { EActFunc } from "./activation-functions"

export interface INet {
    createdAt: Date
    updatedAt: Date
    fcLayers: fcLayers.IFCLayer[]
}

export interface INetConfig {
    neuronsPerHiddenLayer?: number[]
    actFunc?: {
        allLayers?: EActFunc
        hiddenLayers?: EActFunc
        outputLayer?:  EActFunc
        perLayer?: EActFunc[]
    },
    noBias?: boolean
}

/**
 * @typedef ITrainData
 */
export interface ITrainData {
    input:  number[]
    output: number[]
}

/* ------------------------------- Functions -------------------------------- */

/**
 * Creates an untrained neural net with random weights.
 */
export function init(numOfInputs: number, numOfOutputs: number, config?: INetConfig): INet {
    let layers: fcLayers.IFCLayerConfig[] = []

    if (config && config.neuronsPerHiddenLayer ) {
        config.neuronsPerHiddenLayer.forEach((numOfNeurons, i) => {
            layers.push({
                actFunc: 
                    config.actFunc && config.actFunc.perLayer
                    && config.actFunc.perLayer[i] !== undefined
                        ? config.actFunc.perLayer[i] :
                    config.actFunc && config.actFunc.hiddenLayers !== undefined
                        ? config.actFunc.hiddenLayers :
                    config.actFunc && config.actFunc.allLayers !== undefined
                        ? config.actFunc.allLayers
                        : DEFAULT_ACT_FUNC_HIDDEN_LAYER,
                numOfNeurons: numOfNeurons
            })
        })
    }

    layers.push({
        actFunc: 
            config && config.actFunc && config.actFunc.outputLayer !== undefined
                ? config.actFunc.outputLayer :
            config && config.actFunc && config.actFunc.allLayers !== undefined
                ? config.actFunc.allLayers
                : DEFAULT_ACT_FUNC_OUTPUT_LAYER,
        numOfNeurons: numOfOutputs
    })

    return {
        createdAt: new Date,
        updatedAt: new Date,
        fcLayers: fcLayers.init(numOfInputs, layers, config && config.noBias)
    }
}

/**
 * Runs one calculation with a given neural net and one set of input values.
 */
export function calc(net: INet, values: number[]): number[] {
    return fcLayers.calc(net.fcLayers, values)
}

export function train(net: INet, input: number[], expOutput: number[], learnRate?: number): INet {
    learnRate = learnRate || DEFAULT_LEARNING_RATE

    return {
        createdAt: net.createdAt,
        updatedAt: new Date,
        fcLayers: fcLayers.train(net.fcLayers, input, expOutput, learnRate)
    }
}

export function trainSet(net: INet, data:ITrainData[], learnRate?: number): INet {
    const lr = learnRate || DEFAULT_LEARNING_RATE

    let d = deepClone(data)
    d.sort(() => (Math.random() - .5))

    return {
        createdAt: net.createdAt,
        updatedAt: new Date,
        fcLayers: d.reduce(
            (layers, set) => fcLayers.train(layers, set.input, set.output, lr),
            net.fcLayers
        )
    }
}

/* --------------------------------- Intern --------------------------------- */

const DEFAULT_ACT_FUNC_HIDDEN_LAYER = EActFunc.Sigmoid
const DEFAULT_ACT_FUNC_OUTPUT_LAYER = EActFunc.RectifiedLinear
const DEFAULT_LEARNING_RATE = .01