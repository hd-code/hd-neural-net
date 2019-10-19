const assert = require('assert');
const fcl = require('../build/fc-layers');
const actFunc = require('../build/activation-functions');
const helper = require('../build/helper')

const PRECISION = 3 // all calculations should be rounded to 3 digits after comma
/*
describe('Test ' + __filename.substr(__dirname.length), () => {

describe('init()', () => {
    describe('1) 2,2 net', () => {
        let outputLayer = {
            actFunction: fcl.EActFunction.RELU,
            numOfNeurons: 2
        }
        let net = fcl.init(2, outputLayer)

        it('should have 1 layer', () => {
            assert.strictEqual(net.length, 1)
        })
        it('layers actFunc should be ReLU', () => {
            assert.strictEqual(net[0].actFunction, fcl.EActFunction.RELU)
        })
        it('layer should have 3 prevNeurons (bias)', () => {
            assert.strictEqual(net[0].weights.length, 3)
        })
        it('layer should have 2 neurons', () => {
            assert.strictEqual(net[0].weights[0].length, 2)
            assert.strictEqual(net[0].weights[1].length, 2)
            assert.strictEqual(net[0].weights[2].length, 2)
        })
    })

    describe('2) 2,2,2 net', () => {
        let hiddenLayers = [
            {
                actFunction: fcl.EActFunction.SIGMOID,
                numOfNeurons: 2
            }
        ]
        let outputLayer = {
            actFunction: fcl.EActFunction.RELU,
            numOfNeurons: 2
        }
        let net = fcl.init(2, outputLayer, hiddenLayers)

        it('should have 2 layers', () => {
            assert.strictEqual(net.length, 2)
        })

        it('hidden layers actFunc should be Sigmoid', () => {
            assert.strictEqual(net[0].actFunction, fcl.EActFunction.SIGMOID)
        })
        it('hidden layer should have 3 prevNeurons (bias)', () => {
            assert.strictEqual(net[0].weights.length, 3)
        })
        it('hidden layer should have 2 neurons', () => {
            assert.strictEqual(net[0].weights[0].length, 2)
            assert.strictEqual(net[0].weights[1].length, 2)
            assert.strictEqual(net[0].weights[2].length, 2)
        })

        it('output layers actFunc should be ReLU', () => {
            assert.strictEqual(net[1].actFunction, fcl.EActFunction.RELU)
        })
        it('output layer should have 3 prevNeurons (bias)', () => {
            assert.strictEqual(net[1].weights.length, 3)
        })
        it('output layer should have 2 neurons', () => {
            assert.strictEqual(net[1].weights[0].length, 2)
            assert.strictEqual(net[1].weights[1].length, 2)
            assert.strictEqual(net[1].weights[2].length, 2)
        })
    })

    describe('3) 2,2,2 net – no bias', () => {
        let hiddenLayers = [
            {
                actFunction: fcl.EActFunction.SIGMOID,
                numOfNeurons: 2
            }
        ]
        let outputLayer = {
            actFunction: fcl.EActFunction.RELU,
            numOfNeurons: 2
        }
        let net = fcl.init(2, outputLayer, hiddenLayers, true)

        it('should have 2 layers', () => {
            assert.strictEqual(net.length, 2)
        })

        it('hidden layers actFunc should be Sigmoid', () => {
            assert.strictEqual(net[0].actFunction, fcl.EActFunction.SIGMOID)
        })
        it('hidden layer should have 2 neurons', () => {
            assert.strictEqual(net[0].weights.length, 2)
        })
        it('hidden layer should have 2 prevNeurons', () => {
            assert.strictEqual(net[0].weights[0].length, 2)
            assert.strictEqual(net[0].weights[1].length, 2)
        })

        it('output layers actFunc should be ReLU', () => {
            assert.strictEqual(net[1].actFunction, fcl.EActFunction.RELU)
        })
        it('output layer should have 2 prevNeurons', () => {
            assert.strictEqual(net[1].weights.length, 2)
        })
        it('output layer should have 2 neurons', () => {
            assert.strictEqual(net[1].weights[0].length, 2)
            assert.strictEqual(net[1].weights[1].length, 2)
        })
    })

    describe('4) 2,2,3,2 net – all binary as actFunc', () => {
        let hiddenLayers = [
            {
                actFunction: fcl.EActFunction.BINARY,
                numOfNeurons: 2
            },
            {
                actFunction: fcl.EActFunction.BINARY,
                numOfNeurons: 3
            }
        ]
        let outputLayer = {
            actFunction: fcl.EActFunction.BINARY,
            numOfNeurons: 2
        }
        let net = fcl.init(2, outputLayer, hiddenLayers)

        it('should have 3 layers', () => {
            assert.strictEqual(net.length, 3)
        })

        it('hidden layers actFunc should be Binary', () => {
            assert.strictEqual(net[0].actFunction, fcl.EActFunction.BINARY)
            assert.strictEqual(net[1].actFunction, fcl.EActFunction.BINARY)
        })
        it('hidden layers should have 3 prevNeurons (bias)', () => {
            assert.strictEqual(net[0].weights.length, 3)
            assert.strictEqual(net[1].weights.length, 3)
        })
        it('first hidden layers should have 2, second 3 neurons', () => {
            assert.strictEqual(net[0].weights[0].length, 2)
            assert.strictEqual(net[0].weights[1].length, 2)
            assert.strictEqual(net[0].weights[2].length, 2)
            assert.strictEqual(net[1].weights[0].length, 3)
            assert.strictEqual(net[1].weights[1].length, 3)
            assert.strictEqual(net[1].weights[2].length, 3)
        })

        it('output layers actFunc should be Binary', () => {
            assert.strictEqual(net[2].actFunction, fcl.EActFunction.BINARY)
        })
        it('output layer should have 4 prevNeurons (bias)', () => {
            assert.strictEqual(net[2].weights.length, 4)
        })
        it('output layer should have 2 neurons', () => {
            assert.strictEqual(net[2].weights[0].length, 2)
            assert.strictEqual(net[2].weights[1].length, 2)
            assert.strictEqual(net[2].weights[2].length, 2)
            assert.strictEqual(net[2].weights[3].length, 2)
        })
    })
})

describe('calc()', () => {
    describe('1) 2,2 net', () => {
        const NET = [
            {
                actFunction: fcl.EActFunction.RELU,
                weights: [
                    [0.3, 0.7],
                    [0.5, 0.1],
                    [0.2, 2.5]
                ]
            }
        ]

        it('input: 1,1', () => {
            const INPUT  = [1,1] // result: [1, 3.3]
            const OUTPUT = [
                actFunc.apply(1, fcl.EActFunction.RELU),
                actFunc.apply(3.3, fcl.EActFunction.RELU)
            ]
            let result = fcl.calc(INPUT, NET)
            assert.deepStrictEqual(result, OUTPUT)
        })

        it('input: 2,1', () => {
            const INPUT  = [2,1] // result: [1.3, 4]
            const OUTPUT = [
                actFunc.apply(1.3, fcl.EActFunction.RELU),
                actFunc.apply(4, fcl.EActFunction.RELU)
            ]
            let result = fcl.calc(INPUT, NET)
            assert.deepStrictEqual(result, OUTPUT)
        })

        it('input: 2,-5', () => {
            const INPUT  = [2,-5] // result: [-1.7, 3.4]
            const OUTPUT = [
                actFunc.apply(-1.7, fcl.EActFunction.RELU),
                actFunc.apply(3.4, fcl.EActFunction.RELU)
            ]
            let result = fcl.calc(INPUT, NET)
            assert.deepStrictEqual(result, OUTPUT)
        })
    })

    describe('2) 2,2 net – no bias', () => {
        const NET = [
            {
                actFunction: fcl.EActFunction.RELU,
                weights: [
                    [0.3, 0.7],
                    [0.5, 0.1]
                ]
            }
        ]

        it('input: 1,1', () => {
            const INPUT  = [1,1] // result: [0.8, 0.8]
            const OUTPUT = [
                actFunc.apply(0.8, fcl.EActFunction.RELU),
                actFunc.apply(0.8, fcl.EActFunction.RELU)
            ]
            let result = fcl.calc(INPUT, NET)
            result = result.map(num => helper.round(num, PRECISION))
            assert.deepStrictEqual(result, OUTPUT)
        })

        it('input: 2,1', () => {
            const INPUT  = [2,1] // result: [1.1, 1.5]
            const OUTPUT = [
                actFunc.apply(1.1, fcl.EActFunction.RELU),
                actFunc.apply(1.5, fcl.EActFunction.RELU)
            ]
            let result = fcl.calc(INPUT, NET)
            assert.deepStrictEqual(result, OUTPUT)
        })

        it('input: 2,-5', () => {
            const INPUT  = [2,-5] // result: [-1.9, 0.9]
            const OUTPUT = [
                actFunc.apply(-1.9, fcl.EActFunction.RELU),
                actFunc.apply(0.9, fcl.EActFunction.RELU)
            ]
            let result = fcl.calc(INPUT, NET)
            result = result.map(num => helper.round(num, PRECISION))
            assert.deepStrictEqual(result, OUTPUT)
        })
    })

    describe('3) 2,2,2 net – hidden layer linear', () => {
        const NET = [
            {
                actFunction: fcl.EActFunction.LINEAR,
                weights: [
                    [0.3, 0.7],
                    [0.5, 0.1],
                    [0.2, 2.5]
                ]
            },
            {
                actFunction: fcl.EActFunction.RELU,
                weights: [
                    [0.3, 0.7],
                    [0.5, 0.1],
                    [0.2, 2.5]
                ]
            }
        ]

        it('input: 1,1', () => {
            const INPUT  = [1,1] // result hidden layer: [1, 3.3]
                                 // result output layer: [2.15, 3,53]
            const OUTPUT = [
                actFunc.apply(2.15, fcl.EActFunction.RELU),
                actFunc.apply(3.53, fcl.EActFunction.RELU)
            ]
            let result = fcl.calc(INPUT, NET)
            result = result.map(num => helper.round(num, PRECISION))
            assert.deepStrictEqual(result, OUTPUT)
        })

        it('input: 2,1', () => {
            const INPUT  = [2,1] // result hidden layer: [1.3, 4]
                                 // result output layer: [2.59, 3.81]
            const OUTPUT = [
                actFunc.apply(2.59, fcl.EActFunction.RELU),
                actFunc.apply(3.81, fcl.EActFunction.RELU)
            ]
            let result = fcl.calc(INPUT, NET)
            result = result.map(num => helper.round(num, PRECISION))
            assert.deepStrictEqual(result, OUTPUT)
        })

        it('input: 2,-5', () => {
            const INPUT  = [2,-5] // result hidden layer: [-1.7, 3.4]
                                  // result output layer: [1.39, 1.65]
            const OUTPUT = [
                actFunc.apply(1.39, fcl.EActFunction.RELU),
                actFunc.apply(1.65, fcl.EActFunction.RELU)
            ]
            let result = fcl.calc(INPUT, NET)
            result = result.map(num => helper.round(num, PRECISION))
            assert.deepStrictEqual(result, OUTPUT)
        })
    })
})

describe('train()', () => {
    describe('1) 2,2,2 net – hidden layer linear\n' +
             '\t input: 1,1 – output: 1,1.5 – learningRate: 0.1', () => 
    {
        const NET = [
            {
                actFunction: fcl.EActFunction.LINEAR,
                weights: [
                    [0.3, 0.7],
                    [0.5, 0.1],
                    [0.2, 2.5]
                ]
            },
            {
                actFunction: fcl.EActFunction.RELU,
                weights: [
                    [0.3, 0.7],
                    [0.5, 0.1],
                    [0.2, 2.5]
                ]
            }
        ]

        const INPUT  = [1,1]
        const OUTPUT = [1,1.5]
        const LR = .1 // learningRate
        
        // result hidden layer: [1, 3.3],     result output layer: [2.15, 3,53]
        //  delta hidden layer: [1.766, .778], delta output layer: [1.15, 2.03]
        
        const WEIGHTS_HIDDEN = helper.roundMatrix([
            // input * delta * learningRate
            [0.3 - 1 * 1.766 * LR, 0.7 - 1 * .778 * LR],
            [0.5 - 1 * 1.766 * LR, 0.1 - 1 * .778 * LR],
            [0.2 - 1 * 1.766 * LR, 2.5 - 1 * .778 * LR]
        ], PRECISION)
        const WEIGHTS_OUTPUT = helper.roundMatrix([
            // prevLayerOutput * delta * learningRate
            [0.3 -   1 * 1.15 * LR, 0.7 -   1 * 2.03 * LR],
            [0.5 - 3.3 * 1.15 * LR, 0.1 - 3.3 * 2.03 * LR],
            [0.2 -   1 * 1.15 * LR, 2.5 -   1 * 2.03 * LR]
        ], PRECISION)

        let result = fcl.train(INPUT, OUTPUT, LR, NET)

        it('should still have 2 layers', () => {
            assert.strictEqual(result.length, 2)
        })

        it('should still have same activation functions', () => {
            assert.strictEqual(result[0].actFunction, NET[0].actFunction)
            assert.strictEqual(result[1].actFunction, NET[1].actFunction)
        })

        it('should output the corrected weights for both layers', () => {
            let hiddenLayer = helper.roundMatrix(result[0].weights, PRECISION)
            let outputLayer = helper.roundMatrix(result[1].weights, PRECISION)

            assert.deepStrictEqual(hiddenLayer, WEIGHTS_HIDDEN)
            assert.deepStrictEqual(outputLayer, WEIGHTS_OUTPUT)
        })
    })
})

describe('integration tests', () => {
    describe('1) 1,1 net – multiply by 2 with input from 0 to 4', () => {
        const LEARNING_RATE = .1
        const NUM_OF_ITERATIONS = 100

        const TEST_DATA = [
            { input: [0], output: [0] },
            { input: [1], output: [2] },
            { input: [2], output: [4] },
            { input: [3], output: [6] },
            { input: [4], output: [8] },
        ]
        const TEST_DATA_EXTENDED = [
            { input: [5], output: [10] },
            { input: [6], output: [12] },
            { input: [7], output: [14] },
            { input: [8], output: [16] },
            { input: [9], output: [18] },
        ]

        let net = fcl.init(1, {numOfNeurons: 1})
        for (var i = 0, ie = NUM_OF_ITERATIONS; i < ie; i++) {
            for (var j = 0, je = TEST_DATA.length; j < je; j++) {
                net = fcl.train(TEST_DATA[j].input, TEST_DATA[j].output, LEARNING_RATE, net)
            }
        }

        it('should produce decent results after training', () => {
            for (var i = 0, ie = TEST_DATA.length; i < ie; i++) {
                let result = fcl.calc(TEST_DATA[i].input, net)
                result = helper.roundVector(result, 1)
                assert.deepStrictEqual(result, TEST_DATA[i].output)
            }
        })

        it('should produce decent results for data it has not been trained with', () => {
            for (var i = 0, ie = TEST_DATA_EXTENDED.length; i < ie; i++) {
                let result = fcl.calc(TEST_DATA_EXTENDED[i].input, net)
                result = helper.roundVector(result, 1)
                assert.deepStrictEqual(result, TEST_DATA_EXTENDED[i].output)
            }
        })
    })

    describe('2) 2,8,4 net – binary operations AND, OR, XOR, XNOR', () => {
        const LEARNING_RATE = .1
        const NUM_OF_ITERATIONS = 1000

        const TEST_DATA = [
            { input: [0,0], output: [0,0,0,1] },
            { input: [0,1], output: [0,1,1,0] },
            { input: [1,0], output: [0,1,1,0] },
            { input: [1,1], output: [1,1,0,1] },
        ]

        let net = fcl.init(2, {numOfNeurons: 4}, [{numOfNeurons: 8}])
        for (var i = 0, ie = NUM_OF_ITERATIONS; i < ie; i++) {
            for (var j = 0, je = TEST_DATA.length; j < je; j++) {
                net = fcl.train(TEST_DATA[j].input, TEST_DATA[j].output, LEARNING_RATE, net)
            }
        }

        it('should produce decent results after training', () => {
            for (var i = 0, ie = TEST_DATA.length; i < ie; i++) {
                let result = fcl.calc(TEST_DATA[i].input, net)
                result = helper.roundVector(result, 1)
                assert.deepStrictEqual(result, TEST_DATA[i].output)
            }
        })
    })

    describe('3) 2,8,1 net – ackermann function from 0,0 to 2,2', () => {
        const LEARNING_RATE = .1
        const NUM_OF_ITERATIONS = 10000

        const TEST_DATA = [
            { input: [0,0], output: [1] },
            { input: [0,1], output: [2] },
            { input: [0,2], output: [3] },
            { input: [1,0], output: [2] },
            { input: [1,1], output: [3] },
            { input: [1,2], output: [4] },
            { input: [2,0], output: [3] },
            { input: [2,1], output: [5] },
            { input: [2,2], output: [7] },
        ]

        const TEST_DATA_EXTENDED = [
            { input: [0,3], output: [4] },
            { input: [1,3], output: [5] },
            { input: [2,3], output: [9] },
        ]

        let net = fcl.init(2, {numOfNeurons: 1}, [{numOfNeurons: 8}])
        for (var i = 0, ie = NUM_OF_ITERATIONS; i < ie; i++) {
            for (var j = 0, je = TEST_DATA.length; j < je; j++) {
                net = fcl.train(TEST_DATA[j].input, TEST_DATA[j].output, LEARNING_RATE, net)
            }
        }

        it('should produce decent results after training', () => {
            for (var i = 0, ie = TEST_DATA.length; i < ie; i++) {
                let result = fcl.calc(TEST_DATA[i].input, net)
                result = helper.roundVector(result, 1)
                assert.deepStrictEqual(result, TEST_DATA[i].output)
            }
        })

        it('should produce almost decent results after training for unknown data', () => {
            for (var i = 0, ie = TEST_DATA_EXTENDED.length; i < ie; i++) {
                let result = fcl.calc(TEST_DATA_EXTENDED[i].input, net)
                let expOutput = TEST_DATA_EXTENDED[i].output
                assert.ok(expOutput - 1 < result < expOutput + 1)
            }
        })
    })
})

})

*/