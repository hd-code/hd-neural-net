const assert = require('assert');
const { EActFunc, train } = require('../build/main');
const { round } = require('../build/helper');

describe('Test train()', () => {
    describe('1) 1,1 net – multiply by 2', () => {
        const startMultiplier = .8
        const correctMultiplier = 2
        const DATA = { input: [ 2 ], output: [ 4 ] }
        const net = {
            createdAt: new Date,
            updatedAt: new Date,
            fcLayers: [
                { actFunc: EActFunc.Linear,
                  weights: [ [ startMultiplier ] ]
                }
            ]
        }
        
        it('should be closer to correctMultiplier after one training', () => {
            const trained = train(net, DATA.input, DATA.output)
            const newMultiplier = trained.fcLayers[0].weights[0][0]
            assert.ok(newMultiplier > startMultiplier)
        })

        it('should have correctMultiplier after 200 epoches', () => {
            let trainedNet = net;
            for (let i = 0; i < 200; i++) {
                trainedNet = train(trainedNet, DATA.input, DATA.output)
            }
            const newMultiplier = round(trainedNet.fcLayers[0].weights[0][0], 2)
            assert.strictEqual(newMultiplier, correctMultiplier)
        })

        it('should have correctMultiplier after 11 epoches with learnRate = .1', () => {
            let trainedNet = net;
            for (let i = 0; i < 11; i++) {
                trainedNet = train(trainedNet, DATA.input, DATA.output, .1)
            }
            const newMultiplier = round(trainedNet.fcLayers[0].weights[0][0], 2)
            assert.strictEqual(newMultiplier, correctMultiplier)
        })
    })
})