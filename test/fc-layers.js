const assert = require('assert');
const { init, calc, train } = require('../build/fc-layers');
const { EActFunc } = require('../build/activation-functions');
const { roundVector } = require('../build/math');

describe('Test init()', () => {
    describe('1) 2,2 net', () => {
        const layerConfig = [
            { numOfNeurons: 2, actFunc: EActFunc.RectifiedLinear }
        ]
        const layers = init(2, layerConfig)

        it('should have 1 layer', () => {
            assert.strictEqual(layers.length, 1)
        })
        it('layers actFunc should be ReLU', () => {
            assert.strictEqual(layers[0].actFunc, EActFunc.RectifiedLinear)
        })
        it('layer should have 3 prevNeurons (bias)', () => {
            assert.strictEqual(layers[0].weights.length, 3)
        })
        it('layer should have 2 neurons', () => {
            assert.strictEqual(layers[0].weights[0].length, 2)
            assert.strictEqual(layers[0].weights[1].length, 2)
            assert.strictEqual(layers[0].weights[2].length, 2)
        })
    })

    describe('2) 2,2 net – noBias', () => {
        const layerConfig = [
            { numOfNeurons: 2, actFunc: EActFunc.RectifiedLinear }
        ]
        const layers = init(2, layerConfig, true)

        it('should have 1 layer', () => {
            assert.strictEqual(layers.length, 1)
        })
        it('layers actFunc should be ReLU', () => {
            assert.strictEqual(layers[0].actFunc, EActFunc.RectifiedLinear)
        })
        it('layer should have 2 prevNeurons (noBias)', () => {
            assert.strictEqual(layers[0].weights.length, 2)
        })
        it('layer should have 2 neurons', () => {
            assert.strictEqual(layers[0].weights[0].length, 2)
            assert.strictEqual(layers[0].weights[1].length, 2)
        })
    })

    describe('3) 3,7,5,4 net', () => {
        const layerConfig = [
            { numOfNeurons: 7, actFunc: EActFunc.Sigmoid },
            { numOfNeurons: 5, actFunc: EActFunc.Sigmoid },
            { numOfNeurons: 4, actFunc: EActFunc.RectifiedLinear },
        ]
        const layers = init(3, layerConfig)

        it('should have 3 layer', () => {
            assert.strictEqual(layers.length, 3)
        })

        it('hidden layers actFunc should be Sigmoid', () => {
            assert.strictEqual(layers[0].actFunc, EActFunc.Sigmoid)
            assert.strictEqual(layers[1].actFunc, EActFunc.Sigmoid)
        })
        it('hidden layers should have 4 and 8 prevNeurons (bias!)', () => {
            assert.strictEqual(layers[0].weights.length, 4)
            assert.strictEqual(layers[1].weights.length, 8)
        })
        it('hidden layers should have 7 and 5 neurons', () => {
            assert.strictEqual(layers[0].weights[0].length, 7)
            assert.strictEqual(layers[0].weights[1].length, 7)
            assert.strictEqual(layers[0].weights[2].length, 7)
            assert.strictEqual(layers[0].weights[3].length, 7)

            assert.strictEqual(layers[1].weights[0].length, 5)
            assert.strictEqual(layers[1].weights[1].length, 5)
            assert.strictEqual(layers[1].weights[2].length, 5)
            assert.strictEqual(layers[1].weights[3].length, 5)
            assert.strictEqual(layers[1].weights[4].length, 5)
            assert.strictEqual(layers[1].weights[5].length, 5)
            assert.strictEqual(layers[1].weights[6].length, 5)
            assert.strictEqual(layers[1].weights[7].length, 5)
        })

        it('output layers actFunc should be ReLU', () => {
            assert.strictEqual(layers[2].actFunc, EActFunc.RectifiedLinear)
        })
        it('output layer should have 6 prevNeurons (bias!)', () => {
            assert.strictEqual(layers[2].weights.length, 6)
        })
        it('output layer should have 4 neurons', () => {
            assert.strictEqual(layers[2].weights[0].length, 4)
            assert.strictEqual(layers[2].weights[1].length, 4)
            assert.strictEqual(layers[2].weights[2].length, 4)
            assert.strictEqual(layers[2].weights[3].length, 4)
            assert.strictEqual(layers[2].weights[4].length, 4)
            assert.strictEqual(layers[2].weights[5].length, 4)
        })
    })

    describe('4) 3,7,5,4 net – noBias', () => {
        const layerConfig = [
            { numOfNeurons: 7, actFunc: EActFunc.Sigmoid },
            { numOfNeurons: 5, actFunc: EActFunc.Sigmoid },
            { numOfNeurons: 4, actFunc: EActFunc.RectifiedLinear },
        ]
        const layers = init(3, layerConfig, true)

        it('should have 3 layer', () => {
            assert.strictEqual(layers.length, 3)
        })

        it('hidden layers actFunc should be Sigmoid', () => {
            assert.strictEqual(layers[0].actFunc, EActFunc.Sigmoid)
            assert.strictEqual(layers[1].actFunc, EActFunc.Sigmoid)
        })
        it('hidden layers should have 3 and 7 prevNeurons (noBias!)', () => {
            assert.strictEqual(layers[0].weights.length, 3)
            assert.strictEqual(layers[1].weights.length, 7)
        })
        it('hidden layers should have 7 and 5 neurons', () => {
            assert.strictEqual(layers[0].weights[0].length, 7)
            assert.strictEqual(layers[0].weights[1].length, 7)
            assert.strictEqual(layers[0].weights[2].length, 7)

            assert.strictEqual(layers[1].weights[0].length, 5)
            assert.strictEqual(layers[1].weights[1].length, 5)
            assert.strictEqual(layers[1].weights[2].length, 5)
            assert.strictEqual(layers[1].weights[3].length, 5)
            assert.strictEqual(layers[1].weights[4].length, 5)
            assert.strictEqual(layers[1].weights[5].length, 5)
            assert.strictEqual(layers[1].weights[6].length, 5)
        })

        it('output layers actFunc should be ReLU', () => {
            assert.strictEqual(layers[2].actFunc, EActFunc.RectifiedLinear)
        })
        it('output layer should have 5 prevNeurons (noBias!)', () => {
            assert.strictEqual(layers[2].weights.length, 5)
        })
        it('output layer should have 4 neurons', () => {
            assert.strictEqual(layers[2].weights[0].length, 4)
            assert.strictEqual(layers[2].weights[1].length, 4)
            assert.strictEqual(layers[2].weights[2].length, 4)
            assert.strictEqual(layers[2].weights[3].length, 4)
            assert.strictEqual(layers[2].weights[4].length, 4)
        })
    })

    describe('5) 2,2,2,2 net – actFunc: Sigmoid, Linear, Binary', () => {
        const layerConfig = [
            { numOfNeurons: 2, actFunc: EActFunc.Sigmoid },
            { numOfNeurons: 2, actFunc: EActFunc.Linear },
            { numOfNeurons: 2, actFunc: EActFunc.Binary },
        ]
        const layers = init(2, layerConfig)

        it('should have 3 layer', () => {
            assert.strictEqual(layers.length, 3)
        })

        it('hidden layers actFunc should be Sigmoid and Linear', () => {
            assert.strictEqual(layers[0].actFunc, EActFunc.Sigmoid)
            assert.strictEqual(layers[1].actFunc, EActFunc.Linear)
        })
        it('hidden layers should have 3 prevNeurons (Bias!)', () => {
            assert.strictEqual(layers[0].weights.length, 3)
            assert.strictEqual(layers[1].weights.length, 3)
        })
        it('hidden layers should have 2 neurons', () => {
            assert.strictEqual(layers[0].weights[0].length, 2)
            assert.strictEqual(layers[0].weights[1].length, 2)
            assert.strictEqual(layers[0].weights[2].length, 2)

            assert.strictEqual(layers[1].weights[0].length, 2)
            assert.strictEqual(layers[1].weights[1].length, 2)
            assert.strictEqual(layers[1].weights[2].length, 2)
        })

        it('output layers actFunc should be Binary', () => {
            assert.strictEqual(layers[2].actFunc, EActFunc.Binary)
        })
        it('output layer should have 3 prevNeurons (Bias!)', () => {
            assert.strictEqual(layers[2].weights.length, 3)
        })
        it('output layer should have 2 neurons', () => {
            assert.strictEqual(layers[2].weights[0].length, 2)
            assert.strictEqual(layers[2].weights[1].length, 2)
            assert.strictEqual(layers[2].weights[2].length, 2)
        })
    })
})

describe('Test calc()', () => {
    describe('1) 2,2 net – noBias, Linear', () => {
        const net =  [
            {   actFunc: EActFunc.Linear,
                weights: [  [.2, .6],
                            [.5, .3]     ]
            }
        ]

        it('input: [0,0] -> output: [0,0]', () => {
            const data = { input: [0,0], output: [0,0] }
            const result = roundVector(calc(data.input, net), 2)
            assert.deepStrictEqual(result, data.output)
        })

        it('input: [1,1] -> output: [.7,.9]', () => {
            const data = { input: [1,1], output: [.7,.9] }
            const result = roundVector(calc(data.input, net), 2)
            assert.deepStrictEqual(result, data.output)
        })

        it('input: [2,1] -> output: [.9,1.5]', () => {
            const data = { input: [2,1], output: [.9,1.5] }
            const result = roundVector(calc(data.input, net), 2)
            assert.deepStrictEqual(result, data.output)
        })
    })

    describe('2) 2,2 net – bias, Linear', () => {
        const net = [
            {   actFunc: EActFunc.Linear,
                weights: [  [.2, .6],
                            [.5, .3],
                            [.1, .2]     ]
            }
        ]

        it('input: [0,0] -> output: [.1,.2]', () => {
            const data = { input: [0,0], output: [.1,.2] }
            const result = roundVector(calc(data.input, net), 2)
            assert.deepStrictEqual(result, data.output)
        })

        it('input: [1,1] -> output: [.8,1.1]', () => {
            const data = { input: [1,1], output: [.8,1.1] }
            const result = roundVector(calc(data.input, net), 2)
            assert.deepStrictEqual(result, data.output)
        })

        it('input: [2,1] -> output: [1,1.7]', () => {
            const data = { input: [2,1], output: [1,1.7] }
            const result = roundVector(calc(data.input, net), 2)
            assert.deepStrictEqual(result, data.output)
        })
    })

    describe('3) 2,2 net – bias, Sigmoid', () => {
        const net = [
            {   actFunc: EActFunc.Sigmoid,
                weights: [  [.2, .6],
                            [.5, .3],
                            [.1, .2]     ]
            }
        ]

        it('input: [0,0] -> output: [.52,.55]', () => {
            const data = { input: [0,0], output: [.52,.55] }
            const result = roundVector(calc(data.input, net), 2)
            assert.deepStrictEqual(result, data.output)
        })

        it('input: [1,1] -> output: [.69,.75]', () => {
            const data = { input: [1,1], output: [.69,.75] }
            const result = roundVector(calc(data.input, net), 2)
            assert.deepStrictEqual(result, data.output)
        })

        it('input: [2,1] -> output: [.73,.85]', () => {
            const data = { input: [2,1], output: [.73,.85] }
            const result = roundVector(calc(data.input, net), 2)
            assert.deepStrictEqual(result, data.output)
        })
    })

    describe('4) 2,2,2 net – bias, Sigmoid, ReLU', () => {
        const net = [
            {   actFunc: EActFunc.Sigmoid,
                weights: [  [.2, .6],
                            [.5, .3],
                            [.1, .2]     ]
            },
            {   actFunc: EActFunc.RectifiedLinear,
                weights: [  [.2, .6],
                            [.5, .3],
                            [.1, .2]     ]
            }
        ]

        it('input: [0,0] -> output: [.48,.68]', () => {
            const data = { input: [0,0], output: [.48,.68] }
            const result = roundVector(calc(data.input, net), 2)
            assert.deepStrictEqual(result, data.output)
        })

        it('input: [1,1] -> output: [.61,.84]', () => {
            const data = { input: [1,1], output: [.61,.84] }
            const result = roundVector(calc(data.input, net), 2)
            assert.deepStrictEqual(result, data.output)
        })

        it('input: [2,1] -> output: [.67,.89]', () => {
            const data = { input: [2,1], output: [.67,.89] }
            const result = roundVector(calc(data.input, net), 2)
            assert.deepStrictEqual(result, data.output)
        })
    })
})