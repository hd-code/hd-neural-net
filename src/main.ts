import { ACT_FUNC } from "./activation-functions";
import * as fcLayers from "./fc-layers";
import { deepClone } from "./helper";

export interface INet {
    createdAt: Date
    updatedAt: Date
    fcLayers: fcLayers.IFCLayer[]
}

export interface INetConfig {
    neuronsPerHiddenLayer: number[]
    actFunc?: {
        allLayers?: ACT_FUNC
        hiddenLayers?: ACT_FUNC
        outputLayer?:  ACT_FUNC
        perLayer?: ACT_FUNC[]
    },
    noBias?: boolean
}

export function init(numOfInputs: number, numOfOutputs: number, config?: INetConfig): INet {
    let layers: fcLayers.IFCLayerConfig[] = []

    if (config) {
        config.neuronsPerHiddenLayer.forEach((numOfNeurons, i) => {
            layers.push({
                actFunc: !config.actFunc          ? DEFAULT_ACT_FUNC_HIDDEN_LAYER
                    : config.actFunc.perLayer     ? config.actFunc.perLayer[i]
                    : config.actFunc.hiddenLayers ? config.actFunc.hiddenLayers
                    : config.actFunc.allLayers    ? config.actFunc.allLayers
                    : DEFAULT_ACT_FUNC_HIDDEN_LAYER,
                numOfNeurons: numOfNeurons
            })
        })
    }

    layers.push({
        actFunc: !config                  ? DEFAULT_ACT_FUNC_OUTPUT_LAYER
            : !config.actFunc             ? DEFAULT_ACT_FUNC_OUTPUT_LAYER
            :  config.actFunc.outputLayer ? config.actFunc.outputLayer
            :  config.actFunc.allLayers   ? config.actFunc.allLayers
            : DEFAULT_ACT_FUNC_OUTPUT_LAYER,
        numOfNeurons: numOfOutputs
    })

    return {
        createdAt: new Date,
        updatedAt: new Date,
        fcLayers: fcLayers.init(numOfInputs, layers, config && config.noBias)
    }
}

export interface ITrainDataPartial {
    input: number[]
}

export function calc(net: INet, values: number[]): number[] {
    return fcLayers.calc(net.fcLayers, values)
}

export function calcSet(net: INet, data: ITrainDataPartial[]): number[][] {
    return data.map(values => fcLayers.calc(net.fcLayers, values.input))
}

export function train(net: INet, input: number[], expOutput: number[], learnRate?: number): INet {
    learnRate = learnRate || .01

    return {
        createdAt: net.createdAt,
        updatedAt: new Date,
        fcLayers: fcLayers.train(net.fcLayers, input, expOutput, learnRate)
    }
}

export interface ITrainData extends ITrainDataPartial {
    output: number[]
}

export function trainSet(net: INet, data:ITrainData[], learnRate?: number): INet {
    const lr = learnRate || .01

    let d = deepClone(data)
    d.sort(() => Math.random() * 2 - 1)

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

const DEFAULT_ACT_FUNC_HIDDEN_LAYER = ACT_FUNC.Sigmoid
const DEFAULT_ACT_FUNC_OUTPUT_LAYER = ACT_FUNC.RectifiedLinear

/* -------------------------------- Testing --------------------------------- */

const NET = init(2, 4, {
    neuronsPerHiddenLayer: [16]
})

const DATA = [
    { input: [0,0], output: [0,0,0,1] },
    { input: [0,1], output: [0,1,1,0] },
    { input: [1,0], output: [0,1,1,0] },
    { input: [1,1], output: [1,1,0,1] }
]

console.log('Before:', calcSet(NET, DATA));

let newNet = deepClone(NET)
for (let i = 0; i < 10000; i++) {
    newNet = trainSet(newNet, DATA)
}

console.log('After:', calcSet(newNet, DATA));