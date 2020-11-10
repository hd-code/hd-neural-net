import { Activation } from './models/activation';
import * as fcLayers from './models/fc-layers';
import { deepClone } from './helper/clone';

// -----------------------------------------------------------------------------

export { Activation } from './models/activation';

export interface INet {
    createdAt: Date
    updatedAt: Date
    fcLayers: fcLayers.IFCLayer[]
}

export interface INetOptions {
    hiddenLayers?: Activation
    outputLayer?: Activation
    noBias?: boolean
}

export function init(numOfInputs: number, numOfOutputs: number, neuronsPerHiddenLayer?: number[], options?: INetOptions): INet {
    const actFHidden = options && options.hiddenLayers !== undefined
        ? options.hiddenLayers : DEFAULT_ACT_FUNC_HIDDEN_LAYER;
    const actFOutput = options && options.outputLayer !== undefined
        ? options.outputLayer : DEFAULT_ACT_FUNC_OUTPUT_LAYER;
    const noBias = options && options.noBias || false;

    const layers: fcLayers.IFCLayerConfig[] = [];
    neuronsPerHiddenLayer && neuronsPerHiddenLayer.forEach(numOfNeurons => {
        layers.push({
            actFunc: actFHidden,
            numOfNeurons
        });
    });
    layers.push({
        actFunc: actFOutput,
        numOfNeurons: numOfOutputs
    });

    return {
        createdAt: new Date,
        updatedAt: new Date,
        fcLayers: fcLayers.init(numOfInputs, layers, noBias)
    };
}

export function calc(input: number[], net: INet): number[] {
    return fcLayers.calc(input, net.fcLayers);
}

export function train(input: number[], expOutput: number[], net: INet, learnRate?: number): INet {
    return {
        createdAt: net.createdAt,
        fcLayers: fcLayers.train(input, expOutput, net.fcLayers, learnRate||DEFAULT_LEARNING_RATE),
        updatedAt: new Date
    };
}

export interface IDataSet {
    input: number[],
    output: number[]
}

export function trainSet(_data: IDataSet[], net: INet, _learnRate?: number): INet {
    const learnRate = _learnRate || DEFAULT_LEARNING_RATE;

    const data = deepClone(_data);
    data.sort(() => Math.random() - .5);

    const trainedNet = data.reduce(
        (layers, data) => fcLayers.train(data.input, data.output, layers, learnRate),
        net.fcLayers
    );

    return {
        createdAt: net.createdAt,
        updatedAt: new Date,
        fcLayers: trainedNet
    };
}

// -----------------------------------------------------------------------------

const DEFAULT_ACT_FUNC_HIDDEN_LAYER = Activation.Sigmoid;
const DEFAULT_ACT_FUNC_OUTPUT_LAYER = Activation.RectifiedLinear;
const DEFAULT_LEARNING_RATE = 0.01;