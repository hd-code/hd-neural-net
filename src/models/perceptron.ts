import * as Matrix from '../../lib/math/matrix';
import * as Vector from '../../lib/math/vector';
import { getFloat } from '../../lib/math/random';

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
    const output = calc(p, input);
    const error = expected - output;
    const deltaW = input.map(i => error * i);
    const deltaB = error;
    return {
        bias: p.bias + learnRate * deltaB,
        weights: p.weights.map((w,i) => w + learnRate * deltaW[i]),
    };
}

export function trainBatch(p: Perceptron, input: number[][], expected: number[], learnRate: number): Perceptron {
    const delta = calcDeltaBatch(p, input, expected);
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
    const deltaB = error;
    const deltaW = error.map((err, i) => Vector.scale(err, input[i]));
    return {
        bias: Vector.avg(deltaB),
        weights: Matrix.transpose(deltaW).map(Vector.avg),
    };
}