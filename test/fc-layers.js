const assert = require('assert');
const fcl = require('../build/fc-layers');
const actFunc = require('../build/activation-functions');
const types = require('../build/types')



describe('Test ' + __filename.substr(__dirname.length), () => {

describe('init', () => {
    describe('1) 2,2 Net', () => {
        let outputLayer = {
            actFunction: types.EActFunction.RELU,
            numOfNeurons: 2
        }
        let net = fcl.init(2, outputLayer)

        it('should have 1 layer', () => {
            assert.strictEqual(net.length, 1)
        })
        it('layers actFunc should be ReLU', () => {
            assert.strictEqual(net[0].actFunction, types.EActFunction.RELU)
        })
        it('layer should have 2 neurons', () => {
            assert.strictEqual(net[0].weights.length, 2)
        })
        it('both should have 3 weights', () => {
            assert.strictEqual(net[0].weights[0].length, 3)
            assert.strictEqual(net[0].weights[1].length, 3)
        })
    })

    describe('2) 2,2,2 Net', () => {
        let hiddenLayers = [
            {
                actFunction: types.EActFunction.SIGMOID,
                numOfNeurons: 2
            }
        ]
        let outputLayer = {
            actFunction: types.EActFunction.RELU,
            numOfNeurons: 2
        }
        let net = fcl.init(2, outputLayer, hiddenLayers)

        it('should have 2 layers', () => {
            assert.strictEqual(net.length, 2)
        })

        it('hidden layers actFunc should be Sigmoid', () => {
            assert.strictEqual(net[0].actFunction, types.EActFunction.SIGMOID)
        })
        it('hidden layer should have 2 neurons', () => {
            assert.strictEqual(net[0].weights.length, 2)
        })
        it('hidden layers neurons should have 3 weights', () => {
            assert.strictEqual(net[0].weights[0].length, 3)
            assert.strictEqual(net[0].weights[1].length, 3)
        })

        it('output layers actFunc should be ReLU', () => {
            assert.strictEqual(net[1].actFunction, types.EActFunction.RELU)
        })
        it('output layer should have 2 neurons', () => {
            assert.strictEqual(net[1].weights.length, 2)
        })
        it('output layers neurons should have 3 weights', () => {
            assert.strictEqual(net[1].weights[0].length, 3)
            assert.strictEqual(net[1].weights[1].length, 3)
        })
    })

    describe('3) 2,2,2 Net without bias', () => {
        let hiddenLayers = [
            {
                actFunction: types.EActFunction.SIGMOID,
                numOfNeurons: 2
            }
        ]
        let outputLayer = {
            actFunction: types.EActFunction.RELU,
            numOfNeurons: 2
        }
        let net = fcl.init(2, outputLayer, hiddenLayers, true)

        it('should have 2 layers', () => {
            assert.strictEqual(net.length, 2)
        })

        it('hidden layers actFunc should be Sigmoid', () => {
            assert.strictEqual(net[0].actFunction, types.EActFunction.SIGMOID)
        })
        it('hidden layer should have 2 neurons', () => {
            assert.strictEqual(net[0].weights.length, 2)
        })
        it('hidden layers neurons should have 2 weights', () => {
            assert.strictEqual(net[0].weights[0].length, 2)
            assert.strictEqual(net[0].weights[1].length, 2)
        })

        it('output layers actFunc should be ReLU', () => {
            assert.strictEqual(net[1].actFunction, types.EActFunction.RELU)
        })
        it('output layer should have 2 neurons', () => {
            assert.strictEqual(net[1].weights.length, 2)
        })
        it('output layers neurons should have 2 weights', () => {
            assert.strictEqual(net[1].weights[0].length, 2)
            assert.strictEqual(net[1].weights[1].length, 2)
        })
    })

    describe('4) 2,2,3,2 Net all binary as actFunc', () => {
        let hiddenLayers = [
            {
                actFunction: types.EActFunction.BINARY,
                numOfNeurons: 2
            },
            {
                actFunction: types.EActFunction.BINARY,
                numOfNeurons: 3
            }
        ]
        let outputLayer = {
            actFunction: types.EActFunction.BINARY,
            numOfNeurons: 2
        }
        let net = fcl.init(2, outputLayer, hiddenLayers)

        it('should have 3 layers', () => {
            assert.strictEqual(net.length, 3)
        })

        it('hidden layers actFunc should be Binary', () => {
            assert.strictEqual(net[0].actFunction, types.EActFunction.BINARY)
            assert.strictEqual(net[1].actFunction, types.EActFunction.BINARY)
        })
        it('first hidden layer should have 2, second 3 neurons', () => {
            assert.strictEqual(net[0].weights.length, 2)
            assert.strictEqual(net[1].weights.length, 3)
        })
        it('hidden layers neurons should have 3 weights', () => {
            assert.strictEqual(net[0].weights[0].length, 3)
            assert.strictEqual(net[0].weights[1].length, 3)
            assert.strictEqual(net[1].weights[0].length, 3)
            assert.strictEqual(net[1].weights[1].length, 3)
            assert.strictEqual(net[1].weights[2].length, 3)
        })

        it('output layers actFunc should be Binary', () => {
            assert.strictEqual(net[2].actFunction, types.EActFunction.BINARY)
        })
        it('output layer should have 2 neurons', () => {
            assert.strictEqual(net[2].weights.length, 2)
        })
        it('output layers neurons should have 4 weights', () => {
            assert.strictEqual(net[2].weights[0].length, 4)
            assert.strictEqual(net[2].weights[1].length, 4)
        })
    })
})

describe('calc', () => {
    describe('1) 2,2 net', () => {
        const NET = [
            {
                actFunction: types.EActFunction.RELU,
                weights: [
                    [0.3, 0.5, 0.2],
                    [0.7, 0.1, 2.5]
                ]
            }
        ]

        it('1', () => {
            const INPUT  = [1,1] // result: [1, 3.3]
            const OUTPUT = [
                actFunc.apply(1, types.EActFunction.RELU),
                actFunc.apply(3.3, types.EActFunction.RELU)
            ]
            let result = fcl.calc(INPUT, NET)
            assert.deepStrictEqual(result, OUTPUT)
        })

        it('2', () => {
            const INPUT  = [2,1] // result: [1.3, 4]
            const OUTPUT = [
                actFunc.apply(1.3, types.EActFunction.RELU),
                actFunc.apply(4, types.EActFunction.RELU)
            ]
            let result = fcl.calc(INPUT, NET)
            assert.deepStrictEqual(result, OUTPUT)
        })

        it('3', () => {
            const INPUT  = [2,-5] // result: [-1.7, 3.4]
            const OUTPUT = [
                actFunc.apply(-1.7, types.EActFunction.RELU),
                actFunc.apply(3.4, types.EActFunction.RELU)
            ]
            let result = fcl.calc(INPUT, NET)
            assert.deepStrictEqual(result, OUTPUT)
        })
    })

    describe('2) 2,2 net no bias', () => {
        const NET = [
            {
                actFunction: types.EActFunction.RELU,
                weights: [
                    [0.3, 0.5],
                    [0.7, 0.1]
                ]
            }
        ]

        it('1', () => {
            const INPUT  = [1,1] // result: [0.8, 0.8]
            const OUTPUT = [
                actFunc.apply(0.8, types.EActFunction.RELU),
                actFunc.apply(0.8, types.EActFunction.RELU)
            ]
            let result = fcl.calc(INPUT, NET)
            result = result.map(num => round(num, 5))
            assert.deepStrictEqual(result, OUTPUT)
        })

        it('2', () => {
            const INPUT  = [2,1] // result: [1.1, 1.5]
            const OUTPUT = [
                actFunc.apply(1.1, types.EActFunction.RELU),
                actFunc.apply(1.5, types.EActFunction.RELU)
            ]
            let result = fcl.calc(INPUT, NET)
            assert.deepStrictEqual(result, OUTPUT)
        })

        it('3', () => {
            const INPUT  = [2,-5] // result: [-1.9, 0.9]
            const OUTPUT = [
                actFunc.apply(-1.9, types.EActFunction.RELU),
                actFunc.apply(0.9, types.EActFunction.RELU)
            ]
            let result = fcl.calc(INPUT, NET)
            result = result.map(num => round(num, 5))
            assert.deepStrictEqual(result, OUTPUT)
        })
    })

    describe('3) 2,2,2 net hidden layer linear', () => {
        const NET = [
            {
                actFunction: types.EActFunction.LINEAR,
                weights: [
                    [0.3, 0.5, 0.2],
                    [0.7, 0.1, 2.5]
                ]
            },
            {
                actFunction: types.EActFunction.RELU,
                weights: [
                    [0.3, 0.5, 0.2],
                    [0.7, 0.1, 2.5]
                ]
            }
        ]

        it('1', () => {
            const INPUT  = [1,1] // result hidden layer: [1, 3.3]
                                 // result output layer: [2.15, 3,53]
            const OUTPUT = [
                actFunc.apply(2.15, types.EActFunction.RELU),
                actFunc.apply(3.53, types.EActFunction.RELU)
            ]
            let result = fcl.calc(INPUT, NET)
            result = result.map(num => round(num, 5))
            assert.deepStrictEqual(result, OUTPUT)
        })

        it('2', () => {
            const INPUT  = [2,1] // result hidden layer: [1.3, 4]
                                 // result output layer: [2.59, 3.81]
            const OUTPUT = [
                actFunc.apply(2.59, types.EActFunction.RELU),
                actFunc.apply(3.81, types.EActFunction.RELU)
            ]
            let result = fcl.calc(INPUT, NET)
            result = result.map(num => round(num, 5))
            assert.deepStrictEqual(result, OUTPUT)
        })

        it('3', () => {
            const INPUT  = [2,-5] // result hidden layer: [-1.7, 3.4]
                                  // result output layer: [1.39, 1.65]
            const OUTPUT = [
                actFunc.apply(1.39, types.EActFunction.RELU),
                actFunc.apply(1.65, types.EActFunction.RELU)
            ]
            let result = fcl.calc(INPUT, NET)
            result = result.map(num => round(num, 5))
            assert.deepStrictEqual(result, OUTPUT)
        })
    })
})

describe('train', () => {
    describe('1) 2,2,2 net hidden layer linear', () => {
        const NET = [
            {
                actFunction: types.EActFunction.LINEAR,
                weights: [
                    [0.3, 0.5, 0.2],
                    [0.7, 0.1, 2.5]
                ]
            },
            {
                actFunction: types.EActFunction.RELU,
                weights: [
                    [0.3, 0.5, 0.2],
                    [0.7, 0.1, 2.5]
                ]
            }
        ]

        it('1', () => {
            const INPUT  = [1,1]
            const OUTPUT = [1,1.5]
            /*
                result hidden layer: [1, 3.3], result output layer: [2.15, 3,53]
                delta output layer: [1.15, 2.03] delta hidden layer: [1.766, .778]
            */
            // TODO: check plausibility
            fcl.train(INPUT, OUTPUT, 1, NET)
        })
    })
})

})

function round(num, precision) {
    let factor = 10 ** precision
    return Math.round(num * factor) / factor
}