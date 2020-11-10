import * as fcl from '../src/models/fc-layers';
import { Activation } from '../src/models/activation';
import * as fs from 'fs';

// -----------------------------------------------------------------------------
/*
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

console.log('Before:')
data.forEach(data => { console.log(fcl.calc(data.input, net)) })
for (let i = 0, ie = epoches; i < ie; i++) {
    data.forEach(data => {
        net = fcl.train(data.input, data.output, net, learnRate)
    })
}
console.log('After:')
data.forEach(data => { console.log(fcl.calc(data.input, net)) })
fs.writeFile('net.json', JSON.stringify(net), (e) => console.log(e))

// net[0].weights = [ [3.3, 1.5], [3.3, 1.5], [-.6,-2] ]
// net[1].weights = [ [3], [-3], [-.7] ]

// console.log(data[0].input + ':', fcl.calc(data[0].input, net))
// console.log(data[1].input + ':', fcl.calc(data[1].input, net))
// console.log(data[2].input + ':', fcl.calc(data[2].input, net))
// console.log(data[3].input + ':', fcl.calc(data[3].input, net))

// net = fcl.train([1,0], [1], net, .1)
// net = fcl.train([1,0], [1], net, .1)
// console.log(net[0].weights)
// console.log(net[1].weights)

//*/