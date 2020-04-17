const fcl = require('../build/fc-layers')
const {EActFunc} = require('../build/activation-functions')
const fs = require('fs')

const print = console.log


const epoches = 10000
const learnRate = .01

const layers = [
    {actFunc: EActFunc.Sigmoid, numOfNeurons: 2},
    {actFunc: EActFunc.RectifiedLinear, numOfNeurons: 2},
]
let net = fcl.init(2, layers)

const data = [
    {input: [0,0], output: [0,1]},
    {input: [0,1], output: [1,0]},
    {input: [1,0], output: [1,0]},
    {input: [1,1], output: [0,0]},
]

// net[0].weights = [ [3.5, -1], [3.4, -1], [-.4,.8] ]
// net[1].weights = [ [1.7, -2.2], [1.6, .3], [-1.7,1.7] ]

print('Before:')
data.forEach(data => { print(fcl.calc(data.input, net)) })
for (let i = 0, ie = epoches; i < ie; i++) {
    data.forEach(data => {
        net = fcl.train(data.input, data.output, net, learnRate)
    })
}
print('After:')
data.forEach(data => { print(fcl.calc(data.input, net)) })
fs.writeFile('net.json', JSON.stringify(net), (e) => print(e))

// net[0].weights = [ [3.3, 1.5], [3.3, 1.5], [-.6,-2] ]
// net[1].weights = [ [3], [-3], [-.7] ]

// print(data[0].input + ':', fcl.calc(data[0].input, net))
// print(data[1].input + ':', fcl.calc(data[1].input, net))
// print(data[2].input + ':', fcl.calc(data[2].input, net))
// print(data[3].input + ':', fcl.calc(data[3].input, net))

// net = fcl.train([1,0], [1], net, .1)
// net = fcl.train([1,0], [1], net, .1)
// print(net[0].weights)
// print(net[1].weights)