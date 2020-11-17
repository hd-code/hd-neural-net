import { getFloat } from '../helper/random';
import * as Matrix from '../helper/matrix';
import * as Vector from '../helper/vector';

// -----------------------------------------------------------------------------

export interface Perceptron {
    bias: number;
    weights: number[];
}

// -----------------------------------------------------------------------------

export function create(numOfInputs: number): Perceptron {
    return {
        bias: getFloat(),
        weights: getRandomVector(numOfInputs),
    };
}

export function calc(p: Perceptron, input: number[]): number {
    return calcBatch(p, [input] as number[][])[0];
}

export function calcBatch(p: Perceptron, input: number[][]): number[] {
    const weighted = Matrix.mulVector(input, p.weights);
    const biased = weighted.map(x => x + p.bias);
    return binaryVector(biased);
}

export function train(p: Perceptron, input: number[], expected: number, learnRate: number): Perceptron {
    return trainBatch(p, [input], [expected], learnRate);
}

export function trainBatch(p: Perceptron, input: number[][], expected: number[], learnRate: number): Perceptron {
    const delta = calcDeltaBatch(p, input, expected);
    console.log(p);
    return {
        bias: p.bias + learnRate * delta.bias,
        weights: Vector.add(p.weights, Vector.scale(learnRate, delta.weights)),
    };
}

// -----------------------------------------------------------------------------

function getRandomVector(length: number): number[] {
    const result = [];
    for (let i = 0; i < length; i++) {
        result.push(getFloat());
    }
    return result;
}

// -----------------------------------------------------------------------------

function binaryVector(input: number[]): number[] {
    return input.map(binary);
}

function binary(input: number): number {
    return input >= 0 ? 1 : 0;
}

// -----------------------------------------------------------------------------

interface Delta {
    bias: number;
    weights: number[];
}

function calcDeltaBatch(p: Perceptron, input: number[][], expected: number[]): Delta {
    const error = Vector.sub(expected, calcBatch(p, input));
    const deltaB = Vector.scale(p.bias, error);
    const deltaW = error.map(err => Vector.scale(err, p.weights));
    return {
        bias: Vector.avg(deltaB),
        weights: Matrix.transpose(deltaW).map(Vector.avg),
    };
}