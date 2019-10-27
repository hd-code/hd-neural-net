// const assert = require('assert');
// const { init, calc, trainSet } = require('../build/main');
// const { roundVector } = require('../build/helper');

// describe('Test trainSet() – (and all others as well)', () => {
//     describe('1) Binary operations: AND,OR,XOR,XNOR', () => {
//         const net = init(2, 4, { neuronsPerHiddenLayer: [ 4 ] })
        
//         const DATA = [
//             { input: [0,0], output: [0,0,0,1] },
//             { input: [0,1], output: [0,1,1,0] },
//             { input: [1,0], output: [0,1,1,0] },
//             { input: [1,1], output: [1,1,0,1] }
//         ]

//         it('should work after 10000 epoches with learnRate of .1', () => {
//             let trainedNet = net
//             for (let i = 0; i < 10000; i++) {
//                 trainedNet = trainSet(trainedNet, DATA, .01)
//             }
//             DATA.forEach(d => {
//                 const actual = roundVector(calc(trainedNet, d.input), 0)
//                 assert.deepStrictEqual(actual, d.output)
//             })
//         })
//     })
// })