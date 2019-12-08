const fcl = require('./build/fc-layers')
const {EActFunc} = require('./build/activation-functions')

const print = console.log


const epoches = 10000
const learnRate = .1

const layers = [
    {actFunc: EActFunc.Sigmoid, numOfNeurons: 2},
    {actFunc: EActFunc.RectifiedLinear, numOfNeurons: 2},
]
let net = fcl.init(2, layers)

const data = [
    {input: [0,0], output: [0,0]},
    {input: [0,1], output: [1,0]},
    {input: [1,0], output: [1,0]},
    {input: [1,1], output: [0,1]},
]

print('Before:')
data.forEach(data => { print(fcl.calc(data.input, net)) })

for (let i = 0, ie = epoches; i < ie; i++) {
    data.forEach(data => {
        net = fcl.train(data.input, data.output, net, learnRate)
    })
}

print('After:')
data.forEach(data => { print(fcl.calc(data.input, net)) })