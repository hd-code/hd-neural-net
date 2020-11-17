import { getFloat } from '../helper/random';
import { mulVector } from '../helper/matrix';
import { add } from '../helper/vector';

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
    const weighted = mulVector(input, p.weights);
    const biased = weighted.map(x => x + p.bias);
    return binaryVector(biased);
}

export function train(p: Perceptron, input: number[], expected: number, learnRate: number): Perceptron {
    const error = expected - calc(p, input);
    const deltaW = p.weights.map((_, i) => learnRate * error * input[i]);
    const deltaB = learnRate * error * p.bias;
    return {
        bias: p.bias + deltaB,
        weights: add(p.weights, deltaW),
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