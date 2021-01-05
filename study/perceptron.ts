import * as Vector from '../src/helper/vector';

// -----------------------------------------------------------------------------

const cases = [{
    name: 'AND',
    bias: -0.5,
    weights: [1.2,0.3],
    learningRate: 0.1,
    data: [
        { input: [0,0], expected: 0 },
        { input: [0,1], expected: 0 },
        { input: [1,0], expected: 0 },
        { input: [1,1], expected: 1 },
    ],
}, {
    name: 'NOR',
    bias: -0.5,
    weights: [1.2,0.3],
    learningRate: 0.1,
    data: [
        { input: [0,0], expected: 1 },
        { input: [0,1], expected: 0 },
        { input: [1,0], expected: 0 },
        { input: [1,1], expected: 0 },
    ],
}, {
    name: 'only middle',
    bias: 0,
    weights: [-.7,.2,.4],
    learningRate: 0.1,
    data: [
        { input: [0,0,0], expected: 0 },
        { input: [0,0,1], expected: 0 },
        { input: [0,1,0], expected: 1 },
        { input: [0,1,1], expected: 1 },
        { input: [1,0,0], expected: 0 },
        { input: [1,0,1], expected: 0 },
        { input: [1,1,0], expected: 1 },
        { input: [1,1,1], expected: 1 },
    ],
}];

cases.forEach(main);

// -----------------------------------------------------------------------------

function main({data, ...meta}: typeof cases[0]) {
    console.log('\n\n-----------------------------------');
    console.log(meta);
    const tableData = data.map((_, i) => calcData({...meta, data}, i));
    console.table(tableData);
}

function calcData(testCase: typeof cases[0], index: number) {
    const data = testCase.data[index];
    const result = calc(testCase.bias, testCase.weights, data.input);
    const error = data.expected - result;
    const deltaB = error;
    const deltaW = data.input.map(input => input * error);
    return {...data, result, error, deltaB, deltaW};
}

// -----------------------------------------------------------------------------

function calc(bias: number, weights: number[], input: number[]): number {
    const value = Vector.dot(weights, input) + bias;
    return value >= 0 ? 1 : 0;
}