const assert = require('assert');
const main = require('../build/index')
const actFunc = require('../build/activation-functions');

describe('Test ' + __filename.substr(__dirname.length), () => {
    describe('1) 2,2 Net', () => {
        let net = main.init(2,2)
        let layers = net.layers

        it('should have 1 layer', () => {
            assert.strictEqual(layers.length, 1)
        })
        it('layers actFunc should be ReLU', () => {
            assert.strictEqual(layers[0].actFunction, actFunc.EActFunction.RELU)
        })
        it('layer should have 2 neurons', () => {
            assert.strictEqual(layers[0].weights.length, 2)
        })
        it('both should have 3 weights', () => {
            assert.strictEqual(layers[0].weights[0].length, 3)
            assert.strictEqual(layers[0].weights[1].length, 3)
        })
    })

    describe('2) 2,2,2 Net', () => {
        let net = main.init(2,2,[2])
        let layers = net.layers

        it('should have 2 layers', () => {
            assert.strictEqual(layers.length, 2)
        })

        it('hidden layers actFunc should be Sigmoid', () => {
            assert.strictEqual(layers[0].actFunction, actFunc.EActFunction.SIGMOID)
        })
        it('hidden layer should have 2 neurons', () => {
            assert.strictEqual(layers[0].weights.length, 2)
        })
        it('hidden layers neurons should have 3 weights', () => {
            assert.strictEqual(layers[0].weights[0].length, 3)
            assert.strictEqual(layers[0].weights[1].length, 3)
        })

        it('output layers actFunc should be ReLU', () => {
            assert.strictEqual(layers[1].actFunction, actFunc.EActFunction.RELU)
        })
        it('output layer should have 2 neurons', () => {
            assert.strictEqual(layers[1].weights.length, 2)
        })
        it('output layers neurons should have 3 weights', () => {
            assert.strictEqual(layers[1].weights[0].length, 3)
            assert.strictEqual(layers[1].weights[1].length, 3)
        })
    })

    describe('3) 2,2,2 Net without bias', () => {
        let options = {noBias: true}
        let net = main.init(2, 2, [2], options)
        let layers = net.layers

        it('should have 2 layers', () => {
            assert.strictEqual(layers.length, 2)
        })

        it('hidden layers actFunc should be Sigmoid', () => {
            assert.strictEqual(layers[0].actFunction, actFunc.EActFunction.SIGMOID)
        })
        it('hidden layer should have 2 neurons', () => {
            assert.strictEqual(layers[0].weights.length, 2)
        })
        it('hidden layers neurons should have 2 weights', () => {
            assert.strictEqual(layers[0].weights[0].length, 2)
            assert.strictEqual(layers[0].weights[1].length, 2)
        })

        it('output layers actFunc should be ReLU', () => {
            assert.strictEqual(layers[1].actFunction, actFunc.EActFunction.RELU)
        })
        it('output layer should have 2 neurons', () => {
            assert.strictEqual(layers[1].weights.length, 2)
        })
        it('output layers neurons should have 2 weights', () => {
            assert.strictEqual(layers[1].weights[0].length, 2)
            assert.strictEqual(layers[1].weights[1].length, 2)
        })
    })

    describe('4) 2,2,3,2 Net all binary as actFunc', () => {
        let options = {
            activationFunctions: {
                hiddenLayers: [actFunc.EActFunction.BINARY, actFunc.EActFunction.BINARY],
                outputLayer: actFunc.EActFunction.BINARY
            }
        }
        let net = main.init(2, 2, [2, 3], options)
        let layers = net.layers

        it('should have 3 layers', () => {
            assert.strictEqual(layers.length, 3)
        })

        it('hidden layers actFunc should be Binary', () => {
            assert.strictEqual(layers[0].actFunction, actFunc.EActFunction.BINARY)
            assert.strictEqual(layers[1].actFunction, actFunc.EActFunction.BINARY)
        })
        it('first hidden layer should have 2, second 3 neurons', () => {
            assert.strictEqual(layers[0].weights.length, 2)
            assert.strictEqual(layers[1].weights.length, 3)
        })
        it('hidden layers neurons should have 3 weights', () => {
            assert.strictEqual(layers[0].weights[0].length, 3)
            assert.strictEqual(layers[0].weights[1].length, 3)
            assert.strictEqual(layers[1].weights[0].length, 3)
            assert.strictEqual(layers[1].weights[1].length, 3)
            assert.strictEqual(layers[1].weights[2].length, 3)
        })

        it('output layers actFunc should be Binary', () => {
            assert.strictEqual(layers[2].actFunction, actFunc.EActFunction.BINARY)
        })
        it('output layer should have 2 neurons', () => {
            assert.strictEqual(layers[2].weights.length, 2)
        })
        it('output layers neurons should have 4 weights', () => {
            assert.strictEqual(layers[2].weights[0].length, 4)
            assert.strictEqual(layers[2].weights[1].length, 4)
        })
    })
})
/*
describe('calc', () => {
    describe('1) 2,2 layers', () => {
        const NET = [
            {
                actFunction: actFunc.EActFunction.RELU,
                weights: [
                    [0.3, 0.5, 0.2],
                    [0.7, 0.1, 2.5]
                ]
            }
        ]

        it('1', () => {
            const INPUT  = [1,1] // result: [1, 3.3]
            const OUTPUT = [
                actFunc.apply(1, actFunc.EActFunction.RELU),
                actFunc.apply(3.3, actFunc.EActFunction.RELU)
            ]
            let result = fcl.calc(INPUT, NET)
            assert.deepStrictEqual(result, OUTPUT)
        })

        it('2', () => {
            const INPUT  = [2,1] // result: [1.3, 4]
            const OUTPUT = [
                actFunc.apply(1.3, actFunc.EActFunction.RELU),
                actFunc.apply(4, actFunc.EActFunction.RELU)
            ]
            let result = fcl.calc(INPUT, NET)
            assert.deepStrictEqual(result, OUTPUT)
        })

        it('3', () => {
            const INPUT  = [2,-5] // result: [-1.7, 3.4]
            const OUTPUT = [
                actFunc.apply(-1.7, actFunc.EActFunction.RELU),
                actFunc.apply(3.4, actFunc.EActFunction.RELU)
            ]
            let result = fcl.calc(INPUT, NET)
            assert.deepStrictEqual(result, OUTPUT)
        })
    })

    describe('2) 2,2 layers no bias', () => {
        const NET = [
            {
                actFunction: actFunc.EActFunction.RELU,
                weights: [
                    [0.3, 0.5],
                    [0.7, 0.1]
                ]
            }
        ]

        it('1', () => {
            const INPUT  = [1,1] // result: [0.8, 0.8]
            const OUTPUT = [
                actFunc.apply(0.8, actFunc.EActFunction.RELU),
                actFunc.apply(0.8, actFunc.EActFunction.RELU)
            ]
            let result = fcl.calc(INPUT, NET)
            result = result.map(num => round(num, 5))
            assert.deepStrictEqual(result, OUTPUT)
        })

        it('2', () => {
            const INPUT  = [2,1] // result: [1.1, 1.5]
            const OUTPUT = [
                actFunc.apply(1.1, actFunc.EActFunction.RELU),
                actFunc.apply(1.5, actFunc.EActFunction.RELU)
            ]
            let result = fcl.calc(INPUT, NET)
            assert.deepStrictEqual(result, OUTPUT)
        })

        it('3', () => {
            const INPUT  = [2,-5] // result: [-1.9, 0.9]
            const OUTPUT = [
                actFunc.apply(-1.9, actFunc.EActFunction.RELU),
                actFunc.apply(0.9, actFunc.EActFunction.RELU)
            ]
            let result = fcl.calc(INPUT, NET)
            result = result.map(num => round(num, 5))
            assert.deepStrictEqual(result, OUTPUT)
        })
    })

    describe('3) 2,2,2 layers hidden layer linear', () => {
        const NET = [
            {
                actFunction: actFunc.EActFunction.LINEAR,
                weights: [
                    [0.3, 0.5, 0.2],
                    [0.7, 0.1, 2.5]
                ]
            },
            {
                actFunction: actFunc.EActFunction.RELU,
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
                actFunc.apply(2.15, actFunc.EActFunction.RELU),
                actFunc.apply(3.53, actFunc.EActFunction.RELU)
            ]
            let result = fcl.calc(INPUT, NET)
            result = result.map(num => round(num, 5))
            assert.deepStrictEqual(result, OUTPUT)
        })

        it('2', () => {
            const INPUT  = [2,1] // result hidden layer: [1.3, 4]
                                 // result output layer: [2.59, 3.81]
            const OUTPUT = [
                actFunc.apply(2.59, actFunc.EActFunction.RELU),
                actFunc.apply(3.81, actFunc.EActFunction.RELU)
            ]
            let result = fcl.calc(INPUT, NET)
            result = result.map(num => round(num, 5))
            assert.deepStrictEqual(result, OUTPUT)
        })

        it('3', () => {
            const INPUT  = [2,-5] // result hidden layer: [-1.7, 3.4]
                                  // result output layer: [1.39, 1.65]
            const OUTPUT = [
                actFunc.apply(1.39, actFunc.EActFunction.RELU),
                actFunc.apply(1.65, actFunc.EActFunction.RELU)
            ]
            let result = fcl.calc(INPUT, NET)
            result = result.map(num => round(num, 5))
            assert.deepStrictEqual(result, OUTPUT)
        })
    })
})

describe('train', () => {
    describe('1) 2,2,2 layers hidden layer linear', () => {
        const NET = [
            {
                actFunction: actFunc.EActFunction.LINEAR,
                weights: [
                    [0.3, 0.5, 0.2],
                    [0.7, 0.1, 2.5]
                ]
            },
            {
                actFunction: actFunc.EActFunction.RELU,
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
            *
            // TODO: check plausibility
            fcl.train(INPUT, OUTPUT, 1, NET)
        })
    })
})

/*********************************** Helper ***********************************/
function round(number, precision) {
    var pow = 10 ** precision
    return Math.round(number * pow) / pow
}