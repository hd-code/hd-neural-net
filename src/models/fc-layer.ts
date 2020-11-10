import * as Matrix from '../helper/matrix';
import * as Vector from '../helper/vector';
import { Activation, calcActivation, isActivation } from './activation';
import { getFloat } from '../helper/random';

// -----------------------------------------------------------------------------

export interface FCLayer {
    activation: Activation;
    bias: number[];
    weights: number[][]; // [output][input]
}

export function isLayer(layer: any): layer is FCLayer {
    const numOfOutputs: number | undefined = layer?.weights?.length;
    return !!numOfOutputs
        && 'activation' in layer && isActivation(layer.activation)
        && 'bias' in layer && Vector.isVector(layer.bias) && layer.bias.length === numOfOutputs
        && 'weights' in layer && Matrix.isMatrix(layer.weights);
}

export function initLayer(numOfInputs: number, numOfOutputs: number, activation: Activation): FCLayer {
    return {
        activation,
        bias: [...Array(numOfOutputs)].map(() => getFloat()),
        weights: [...Array(numOfOutputs)].map(() => [...Array(numOfInputs)].map(() => getFloat())),
    };
}

export function calcLayer(layer: FCLayer, input: number[]): number[] {
    const weighted = Matrix.mulVector(layer.weights, input);
    const withBias = Vector.add(weighted, layer.bias);
    return calcActivation(layer.activation, withBias);
}

// export function trainLayer(layer: Layer, input: number[])
