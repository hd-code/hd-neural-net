import * as assert from 'assert';
import { calcLayer, initLayer, isLayer } from '../../src/models/fc-layer';
import { Activation } from '../../src/models/activation';

// -----------------------------------------------------------------------------

describe('models/layer', () => {
    describe(isLayer.name, () => {
        [
            {
                name: 'correct layer',
                input: {
                    activation: Activation.Sigmoid,
                    bias: [0.2, 0.3],
                    weights: [ [ 0.1, 0.2 ], [ 0.4, 0.3 ] ],
                },
                expected: true,
            },
            {
                name: 'invalid activation',
                input: {
                    activation: -3,
                    bias: [0.2, 0.3],
                    weights: [ [ 0.1, 0.2 ], [ 0.4, 0.3 ] ],
                },
                expected: false,
            },
            {
                name: 'bias has wrong number of outputs',
                input: {
                    activation: Activation.Sigmoid,
                    bias: [0.2],
                    weights: [ [ 0.1, 0.2 ], [ 0.4, 0.3 ] ],
                },
                expected: false,
            },
            {
                name: 'weights is no matrix',
                input: {
                    activation: Activation.Sigmoid,
                    bias: [0.2, 0.3],
                    weights: [ [ 0.1 ], [ 0.4, 0.3 ] ],
                },
                expected: false,
            },
        ].forEach(({name, input, expected}) => it(name, () => {
            const actual = isLayer(input);
            assert.strictEqual(actual, expected);
        }));
    });

    it(initLayer.name, () => {
        const layer = initLayer(3, 2, Activation.Sigmoid);
        assert.strictEqual(layer.activation, Activation.Sigmoid);
        assert.strictEqual(layer.bias.length, 2);
        assert.strictEqual(layer.weights.length, 2);
        assert.strictEqual(layer.weights[0].length, 3);
    });

    describe(calcLayer.name, () => {
        [
            {
                name: 'should sum up values correctly with linear activation',
                layer: {
                    activation: Activation.Linear,
                    bias: [0.2, 0.3],
                    weights: [ [ 0.1, 0.2 ], [ 0.4, 0.3 ] ],
                },
                input: [1, 1],
                expected: [0.5, 1],
            },
            {
                name: 'should return bias with 0 as input and linear activation',
                layer: {
                    activation: Activation.Linear,
                    bias: [0.2, 0.3],
                    weights: [ [ 0.1, 0.2 ], [ 0.4, 0.3 ] ],
                },
                input: [0, 0],
                expected: [0.2, 0.3],
            },
        ].forEach(({name, layer, input, expected}) => it(name, () => {
            const actual = calcLayer(layer, input);
            assert.deepStrictEqual(actual, expected);
        }));
    });
});