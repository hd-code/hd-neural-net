import * as assert from 'assert';
import * as p from '../../src/models/perceptron';

// -----------------------------------------------------------------------------

describe('perceptron', () => {
    const cases = [
        {
            name: 'AND perceptron',
            learningRate: 0.1,
            perceptron: { bias: -0.5, weights: [1.2,0.3] },
            data: [
                { input: [0,0], result: 0, target: 0, error: 0, deltaW: [ 0,0], deltaB: 0 },
                { input: [0,1], result: 0, target: 0, error: 0, deltaW: [ 0,0], deltaB: 0 },
                { input: [1,0], result: 1, target: 0, error:-1, deltaW: [-1,0], deltaB:-1 },
                { input: [1,1], result: 1, target: 1, error: 0, deltaW: [ 0,0], deltaB: 0 },
            ]
        },
        {
            name: 'NAND perceptron',
            learningRate: 0.1,
            perceptron: { bias: -0.5, weights: [1.2,0.3] },
            data: [
                { input: [0,0], result: 0, target: 1, error: 1, deltaW: [ 0, 0], deltaB: 1 },
                { input: [0,1], result: 0, target: 0, error: 0, deltaW: [ 0, 0], deltaB: 0 },
                { input: [1,0], result: 1, target: 0, error:-1, deltaW: [-1, 0], deltaB:-1 },
                { input: [1,1], result: 1, target: 0, error:-1, deltaW: [-1,-1], deltaB:-1 },
            ]
        },
    ];

    describe(p.calc.name + '()', () => cases.forEach(({name, perceptron, data}) => {
        it(name, () => data.forEach(({input,result: expected}) => {
            assert.strictEqual(p.calc(perceptron, input), expected, 'failed for ' + input);
        }));
    }));

    describe(p.calcBatch.name + '()', () => cases.forEach(({name, perceptron, data}) => {
        it(name, () => {
            const input = data.map(d => d.input);
            const expected = data.map(d => d.result);
            const actual = p.calcBatch(perceptron, input);
            assert.deepStrictEqual(actual, expected);
        });
    }));

    describe(p.train.name + '()', () => cases.forEach(({name, learningRate, perceptron, data}) => {
        it(name, () => data.forEach(d => {
            const expected = {
                bias: perceptron.bias + learningRate * d.deltaB,
                weights: perceptron.weights.map((w,i) => w + learningRate * d.deltaW[i]),
            };
            const actual = p.train(perceptron, d.input, d.target, learningRate);
            assert.deepStrictEqual(actual, expected);
        }));
    }));

    describe(p.trainBatch.name + '()', () => cases.forEach(({name, learningRate, perceptron, data}) => {
        it(name, () => {
            const deltaB = data.reduce((sum, d) => sum + d.deltaB, 0) / data.length;
            const deltaW = data[0].deltaW.map((_,i) => data.reduce((sum, d) => sum + d.deltaW[i], 0) / data.length);
            const expected = {
                bias: perceptron.bias + learningRate * deltaB,
                weights: perceptron.weights.map((w,i) => w + learningRate * deltaW[i]),
            };

            const input = data.map(d => d.input);
            const target = data.map(d => d.target);
            const actual = p.trainBatch(perceptron, input, target, learningRate);

            assert.deepStrictEqual(actual, expected);
        });
    }));
});