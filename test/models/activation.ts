import * as assert from 'assert';
import { Activation, calcActivation, isActivation } from '../../src/models/activation';
import round from '../../src/helper/round';

// -----------------------------------------------------------------------------

describe('models/layer', () => {
    describe(isActivation.name, () => {
        it('should return true for all enum values', () => {
            for (const key in Activation) {
                assert.ok(isActivation(Activation[key]), 'failed for ' + key);
            }
        });

        it('should return false for anything else', () => {
            ['string', -90, 2.5, 99999, [], {}, true
            ].forEach(value => assert.ok(!isActivation(value), 'failed for ' + value));
        });
    });

    describe(calcActivation.name, () => {
        [
            {
                activation: Activation.Linear,
                input: [-10, -1, -0.5, 0, 0.5, 1, 10],
                derivative: false,
                expected: [-10, -1, -0.5, 0, 0.5, 1, 10],
            },
            {
                activation: Activation.Linear,
                input: [-10, -1, -0.5, 0, 0.5, 1, 10],
                derivative: true,
                expected: [1, 1, 1, 1, 1, 1, 1],
            },
            {
                activation: Activation.Sigmoid,
                input: [-10, -1, -0.5, 0, 0.5, 1, 10],
                derivative: false,
                expected: [0.0000453979, 0.268941, 0.377541, 0.5, 0.622459, 0.731059, 0.9999545],
            },
            {
                activation: Activation.Sigmoid,
                input: [-10, -1, -0.5, 0, 0.5, 1, 10],
                derivative: true,
                expected: [0.0000453958, 0.196612, 0.235004, 0.25, 0.235004, 0.196612, 0.0000453958],
            },
            {
                activation: Activation.Tanh,
                input: [-10, -1, -0.5, 0, 0.5, 1, 10],
                derivative: false,
                expected: [-1., -0.761594, -0.462117, 0, 0.462117, 0.761594, 1.],
            },
            {
                activation: Activation.Tanh,
                input: [-10, -1, -0.5, 0, 0.5, 1, 10],
                derivative: true,
                expected: [0, 0.419974, 0.786448, 1, 0.786448, 0.419974, 0],
            },
            // {
            //     activation: Activation.HardTanh,
            //     input: [-10, -1, -0.5, 0, 0.5, 1, 10],
            //     derivative: false,
            //     expected: [-10, -1, -0.5, 0, 0.5, 1, 10],
            // },
            // {
            //     activation: Activation.HardTanh,
            //     input: [-10, -1, -0.5, 0, 0.5, 1, 10],
            //     derivative: true,
            //     expected: [1, 1, 1, 1, 1, 1, 1],
            // },
            {
                activation: Activation.RectifiedLinear,
                input: [-10, -1, -0.5, 0, 0.5, 1, 10],
                derivative: false,
                expected: [0, 0, 0, 0, 0.5, 1, 10],
            },
            {
                activation: Activation.RectifiedLinear,
                input: [-10, -1, -0.5, 0, 0.5, 1, 10],
                derivative: true,
                expected: [0, 0, 0, 0, 1, 1, 1],
            },
            {
                activation: Activation.LeakyRectifiedLinear,
                input: [-10, -1, -0.5, 0, 0.5, 1, 10],
                derivative: false,
                expected: [-0.1, -0.01, -0.005, 0, 0.5, 1, 10],
            },
            {
                activation: Activation.LeakyRectifiedLinear,
                input: [-10, -1, -0.5, 0, 0.5, 1, 10],
                derivative: true,
                expected: [0.01, 0.01, 0.01, 0.01, 1, 1, 1],
            },
            // {
            //     activation: Activation.SoftPlus,
            //     input: [-10, -1, -0.5, 0, 0.5, 1, 10],
            //     derivative: false,
            //     expected: [-10, -1, -0.5, 0, 0.5, 1, 10],
            // },
            // {
            //     activation: Activation.SoftPlus,
            //     input: [-10, -1, -0.5, 0, 0.5, 1, 10],
            //     derivative: true,
            //     expected: [1, 1, 1, 1, 1, 1, 1],
            // },
            // {
            //     activation: Activation.Softmax,
            //     input: [-10, -1, -0.5, 0, 0.5, 1, 10],
            //     derivative: false,
            //     expected: [-10, -1, -0.5, 0, 0.5, 1, 10],
            // },
            // {
            //     activation: Activation.Softmax,
            //     input: [-10, -1, -0.5, 0, 0.5, 1, 10],
            //     derivative: true,
            //     expected: [1, 1, 1, 1, 1, 1, 1],
            // },
            // {
            //     activation: Activation.Binary,
            //     input: [-10, -1, -0.5, 0, 0.5, 1, 10],
            //     derivative: false,
            //     expected: [-10, -1, -0.5, 0, 0.5, 1, 10],
            // },
            // {
            //     activation: Activation.Binary,
            //     input: [-10, -1, -0.5, 0, 0.5, 1, 10],
            //     derivative: true,
            //     expected: [1, 1, 1, 1, 1, 1, 1],
            // },
        ].forEach(({activation, input, derivative, expected}) => it(Activation[activation] + (derivative ? ' derivative':''), () => {
            const precision = 5;
            const actual = calcActivation(activation, input, derivative);
            assert.deepStrictEqual(round(actual, precision), round(expected, precision));
        }));
    });
});