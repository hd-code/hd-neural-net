const assert = require('assert');
const { EActFunc, calc } = require('../build/main');
const { roundVector } = require('../build/helper');

describe('Test calc()', () => {
    describe('1) 2,2 net – noBias, Linear', () => {
        const net = {
            createdAt: new Date,
            updatedAt: new Date,
            fcLayers: [
                { actFunc: EActFunc.Linear,
                  weights: [    [.2, .6],
                                [.5, .3]     ]
                }
            ]
        }

        it('input: [0,0] -> output: [0,0]', () => {
            const data = { input: [0,0], output: [0,0] }
            const result = roundVector(calc(net, data.input), 2)
            assert.deepStrictEqual(result, data.output)
        })

        it('input: [1,1] -> output: [.7,.9]', () => {
            const data = { input: [1,1], output: [.7,.9] }
            const result = roundVector(calc(net, data.input), 2)
            assert.deepStrictEqual(result, data.output)
        })

        it('input: [2,1] -> output: [.9,1.5]', () => {
            const data = { input: [2,1], output: [.9,1.5] }
            const result = roundVector(calc(net, data.input), 2)
            assert.deepStrictEqual(result, data.output)
        })
    })

    describe('2) 2,2 net – bias, Linear', () => {
        const net = {
            createdAt: new Date,
            updatedAt: new Date,
            fcLayers: [
                { actFunc: EActFunc.Linear,
                  weights: [    [.2, .6],
                                [.5, .3],
                                [.1, .2]     ]
                }
            ]
        }

        it('input: [0,0] -> output: [.1,.2]', () => {
            const data = { input: [0,0], output: [.1,.2] }
            const result = roundVector(calc(net, data.input), 2)
            assert.deepStrictEqual(result, data.output)
        })

        it('input: [1,1] -> output: [.8,1.1]', () => {
            const data = { input: [1,1], output: [.8,1.1] }
            const result = roundVector(calc(net, data.input), 2)
            assert.deepStrictEqual(result, data.output)
        })

        it('input: [2,1] -> output: [1,1.7]', () => {
            const data = { input: [2,1], output: [1,1.7] }
            const result = roundVector(calc(net, data.input), 2)
            assert.deepStrictEqual(result, data.output)
        })
    })

    describe('3) 2,2 net – bias, Sigmoid', () => {
        const net = {
            createdAt: new Date,
            updatedAt: new Date,
            fcLayers: [
                { actFunc: EActFunc.Sigmoid,
                  weights: [    [.2, .6],
                                [.5, .3],
                                [.1, .2]     ]
                }
            ]
        }

        it('input: [0,0] -> output: [.52,.55]', () => {
            const data = { input: [0,0], output: [.52,.55] }
            const result = roundVector(calc(net, data.input), 2)
            assert.deepStrictEqual(result, data.output)
        })

        it('input: [1,1] -> output: [.69,.75]', () => {
            const data = { input: [1,1], output: [.69,.75] }
            const result = roundVector(calc(net, data.input), 2)
            assert.deepStrictEqual(result, data.output)
        })

        it('input: [2,1] -> output: [.73,.85]', () => {
            const data = { input: [2,1], output: [.73,.85] }
            const result = roundVector(calc(net, data.input), 2)
            assert.deepStrictEqual(result, data.output)
        })
    })

    describe('4) 2,2,2 net – bias, Sigmoid, ReLU', () => {
        const net = {
            createdAt: new Date,
            updatedAt: new Date,
            fcLayers: [
                { actFunc: EActFunc.Sigmoid,
                  weights: [    [.2, .6],
                                [.5, .3],
                                [.1, .2]     ]
                },
                { actFunc: EActFunc.RectifiedLinear,
                  weights: [    [.2, .6],
                                [.5, .3],
                                [.1, .2]     ]
                }
            ]
        }

        it('input: [0,0] -> output: [.48,.68]', () => {
            const data = { input: [0,0], output: [.48,.68] }
            const result = roundVector(calc(net, data.input), 2)
            assert.deepStrictEqual(result, data.output)
        })

        it('input: [1,1] -> output: [.61,.84]', () => {
            const data = { input: [1,1], output: [.61,.84] }
            const result = roundVector(calc(net, data.input), 2)
            assert.deepStrictEqual(result, data.output)
        })

        it('input: [2,1] -> output: [.67,.89]', () => {
            const data = { input: [2,1], output: [.67,.89] }
            const result = roundVector(calc(net, data.input), 2)
            assert.deepStrictEqual(result, data.output)
        })
    })
})