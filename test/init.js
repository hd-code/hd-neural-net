const assert = require('assert');
const { EActFunc, init } = require('../build/main');

describe('Test init()', () => {
    describe('1) 2,2 net', () => {
        const net = init(2, 2)

        it('should have createdAt', () => {
            assert.ok(net.createdAt instanceof Date)
        })
        it('should have updatedAt', () => {
            assert.ok(net.updatedAt instanceof Date)
        })
        it('createdAt and updatedAt should be equal', () => {
            assert.ok(net.createdAt.getDate() === net.updatedAt.getDate())
        })

        const layers = net.fcLayers

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
        const net = init(2, 2, { noBias: true })

        it('should have createdAt', () => {
            assert.ok(net.createdAt instanceof Date)
        })
        it('should have updatedAt', () => {
            assert.ok(net.updatedAt instanceof Date)
        })
        it('createdAt and updatedAt should be equal', () => {
            assert.ok(net.createdAt.getDate() === net.updatedAt.getDate())
        })

        const layers = net.fcLayers

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
        const net = init(3, 4 , { neuronsPerHiddenLayer: [ 7, 5 ] })

        it('should have createdAt', () => {
            assert.ok(net.createdAt instanceof Date)
        })
        it('should have updatedAt', () => {
            assert.ok(net.updatedAt instanceof Date)
        })
        it('createdAt and updatedAt should be equal', () => {
            assert.ok(net.createdAt.getDate() === net.updatedAt.getDate())
        })

        const layers = net.fcLayers

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
        const net = init(3, 4 , { neuronsPerHiddenLayer: [ 7, 5 ], noBias: true })

        it('should have createdAt', () => {
            assert.ok(net.createdAt instanceof Date)
        })
        it('should have updatedAt', () => {
            assert.ok(net.updatedAt instanceof Date)
        })
        it('createdAt and updatedAt should be equal', () => {
            assert.ok(net.createdAt.getDate() === net.updatedAt.getDate())
        })

        const layers = net.fcLayers

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

    describe('5) 2,2,2,2 net – different actFunc', () => {
        it('allLayers: ReLU', () => {
            const config = {
                neuronsPerHiddenLayer: [ 2, 2 ],
                actFunc: { allLayers: EActFunc.RectifiedLinear }
            }
            const net = init(2, 2, config)

            assert.strictEqual(net.fcLayers[0].actFunc, EActFunc.RectifiedLinear)
            assert.strictEqual(net.fcLayers[1].actFunc, EActFunc.RectifiedLinear)
            assert.strictEqual(net.fcLayers[2].actFunc, EActFunc.RectifiedLinear)
        })

        it('outputLayer: Binary', () => {
            const config = {
                neuronsPerHiddenLayer: [ 2, 2 ],
                actFunc: { outputLayer: EActFunc.Binary }
            }
            const net = init(2, 2, config)

            assert.strictEqual(net.fcLayers[0].actFunc, EActFunc.Sigmoid)
            assert.strictEqual(net.fcLayers[1].actFunc, EActFunc.Sigmoid)
            assert.strictEqual(net.fcLayers[2].actFunc, EActFunc.Binary)
        })

        it('hiddenLayers: HardTanh', () => {
            const config = {
                neuronsPerHiddenLayer: [ 2, 2 ],
                actFunc: { hiddenLayers: EActFunc.HardTanh }
            }
            const net = init(2, 2, config)

            assert.strictEqual(net.fcLayers[0].actFunc, EActFunc.HardTanh)
            assert.strictEqual(net.fcLayers[1].actFunc, EActFunc.HardTanh)
            assert.strictEqual(net.fcLayers[2].actFunc, EActFunc.RectifiedLinear)
        })

        it('hiddenLayers: HardTanh, outputLayer: Linear', () => {
            const config = {
                neuronsPerHiddenLayer: [ 2, 2 ],
                actFunc: {
                    hiddenLayers: EActFunc.HardTanh,
                    outputLayer:  EActFunc.Linear
                }
            }
            const net = init(2, 2, config)

            assert.strictEqual(net.fcLayers[0].actFunc, EActFunc.HardTanh)
            assert.strictEqual(net.fcLayers[1].actFunc, EActFunc.HardTanh)
            assert.strictEqual(net.fcLayers[2].actFunc, EActFunc.Linear)
        })

        it('allLayers: HardTanh, outputLayer: Linear', () => {
            const config = {
                neuronsPerHiddenLayer: [ 2, 2 ],
                actFunc: {
                    allLayers:   EActFunc.HardTanh,
                    outputLayer: EActFunc.Linear
                }
            }
            const net = init(2, 2, config)

            assert.strictEqual(net.fcLayers[0].actFunc, EActFunc.HardTanh)
            assert.strictEqual(net.fcLayers[1].actFunc, EActFunc.HardTanh)
            assert.strictEqual(net.fcLayers[2].actFunc, EActFunc.Linear)
        })

        it('allLayers: HardTanh, hiddenLayers: Linear', () => {
            const config = {
                neuronsPerHiddenLayer: [ 2, 2 ],
                actFunc: {
                    allLayers:    EActFunc.HardTanh,
                    hiddenLayers: EActFunc.Linear
                }
            }
            const net = init(2, 2, config)

            assert.strictEqual(net.fcLayers[0].actFunc, EActFunc.Linear)
            assert.strictEqual(net.fcLayers[1].actFunc, EActFunc.Linear)
            assert.strictEqual(net.fcLayers[2].actFunc, EActFunc.HardTanh)
        })

        it('allLayers: HardTanh, hiddenLayers: Linear, outputLayer: Binary', () => {
            const config = {
                neuronsPerHiddenLayer: [ 2, 2 ],
                actFunc: {
                    allLayers:    EActFunc.HardTanh,
                    hiddenLayers: EActFunc.Linear,
                    outputLayer:  EActFunc.Binary
                }
            }
            const net = init(2, 2, config)

            assert.strictEqual(net.fcLayers[0].actFunc, EActFunc.Linear)
            assert.strictEqual(net.fcLayers[1].actFunc, EActFunc.Linear)
            assert.strictEqual(net.fcLayers[2].actFunc, EActFunc.Binary)
        })
    })
})