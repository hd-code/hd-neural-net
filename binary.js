const fcl = require('./build/main')
const fs = require('fs')

const DATA = [
    { input: [0,0], output: [0] },
    { input: [0,1], output: [1] },
    { input: [1,0], output: [1] },
    { input: [1,1], output: [0] },
]

let net = fcl.init(2, 1, {neuronsPerHiddenLayer: [4, 4]})

for (let i = 0, ie = 10000; i < ie; i++) {
    net = fcl.trainSet(net, DATA, 0.1)
}

console.log('input:', DATA[0].input, 'output:', fcl.calc(net, DATA[0].input))
console.log('input:', DATA[1].input, 'output:', fcl.calc(net, DATA[1].input))
console.log('input:', DATA[2].input, 'output:', fcl.calc(net, DATA[2].input))
console.log('input:', DATA[3].input, 'output:', fcl.calc(net, DATA[3].input))

console.log(net.fcLayers[0].weights)
console.log(net.fcLayers[1].weights)