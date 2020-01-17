import { isKeyOfObject } from '../lib/hd-helper';
import { EActFunction, isActFunction, applyToVector } from './ActFunction';
import * as Vector from './Vector';
import * as Matrix from './Matrix';

// -----------------------------------------------------------------------------

export interface IFcLayer {
    actFunction: EActFunction;
    bias:    Vector.TVector;
    weights: Matrix.TMatrix;
}

export function isFcLayer(layer: any): layer is IFcLayer {
    return isKeyOfObject(layer, 'actFunction', isActFunction)
        && isKeyOfObject(layer, 'bias', Vector.isVector)
        && isKeyOfObject(layer, 'weights', Matrix.isMatrix)
        && layer.bias.length === layer.weights[0].length;
}

export function getNumOfIn_Output(layer: IFcLayer): {input: number, output: number} {
    return {input: layer.weights.length, output: layer.weights[0].length};
}

export function create(numOfInputs: number, numOfOutputs: number, actFunction: EActFunction): IFcLayer {
    return {
        actFunction,
        bias:    Vector.createRandom(numOfOutputs),
        weights: Matrix.createRandom(numOfInputs, numOfOutputs)
    };
}

export function calc(layer: IFcLayer, input: Vector.TVector): Vector.TVector {
    return applyToVector(calcWeighted(layer, input), layer.actFunction);
}

export function train(layer: IFcLayer, input: Vector.TVector, error: Vector.TVector, learnRate: number): {delta: Vector.TVector, layer: IFcLayer} {
    const layerInput = calcWeighted(layer, input);
    const derivative = applyToVector(layerInput, layer.actFunction, true);
    const delta = Vector.mul(derivative, error);

    const deltaWeights = layer.weights.map((_,i) => {
        return Vector.scale(input[i], delta);
    });

    return {
        delta,
        layer: {
            actFunction: layer.actFunction,
            bias:    Vector.sub(layer.bias,    Vector.scale(learnRate, delta)),
            weights: Matrix.sub(layer.weights, Matrix.scale(learnRate, deltaWeights))
        }
    };
}

// -----------------------------------------------------------------------------

function calcWeighted(layer: IFcLayer, input: Vector.TVector): Vector.TVector {
    const weighted = Matrix.mulVector(input, layer.weights);
    return Vector.add(weighted, layer.bias);
}