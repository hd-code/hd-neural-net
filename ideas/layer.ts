import * as Matrix from '../src/helper/matrix';
import * as Vector from '../src/helper/vector';
import { Activation, calcActivation, isActivation } from '../src/models/activation';

// -----------------------------------------------------------------------------

export interface Layer {
    activation: Activation;
    bias: number[];
    weights: number[][]; // [output][input]
}

export function isLayer(layer: any): layer is Layer {
    const numOfOutputs: number | undefined = layer?.weights?.length;
    return !!numOfOutputs
        && 'activation' in layer && isActivation(layer.activation)
        && 'bias' in layer && Vector.isVector(layer.bias) && layer.bias.length === numOfOutputs
        && 'weights' in layer && Matrix.isMatrix(layer.weights);
}

export function initLayer(numOfInputs: number, numOfOutputs: number, activation: Activation): Layer {
    return {
        activation,
        bias: [...Array(numOfOutputs)].map(() => Math.random()),
        weights: [...Array(numOfOutputs)].map(() => [...Array(numOfInputs)].map(() => Math.random())),
    };
}

export function calcLayer(layer: Layer, input: number[]): number[] {
    const weighted = Matrix.mulVector(layer.weights, input);
    const withBias = Vector.add(weighted, layer.bias);
    return calcActivation(layer.activation, withBias);
}

export function trainLayer(layer: Layer, input: number[], error: number[], learnRate: number): Layer {
    const derivative = calcActivation(layer.activation, input, true);
    
    const deltaBias = Vector.mul(error, derivative);
    // ...
}
