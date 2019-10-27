const { EActFunc, init, calc, train, trainSet } = require('./build/main');
const { roundVector } = require('./build/helper');

let net1 = init(2, 2)
let net2 = init(2, 2, {neuronsPerHiddenLayer:[2]})
net1.fcLayers[0].actFunc = EActFunc.Linear
net1.fcLayers[0].weights = [
    [.2, .4],
    [.3, .5],
    [.5, .1],
]
net2.fcLayers[0].actFunc = EActFunc.Linear
net2.fcLayers[0].weights = [
    [.3, .5],
    [.5, .1],
    [.2, .4],
]

net2.fcLayers[1].actFunc = EActFunc.Linear
net2.fcLayers[1].weights = [
    [.2, .4],
    [.3, .5],
    [.5, .1],
]

console.log('net1:');
console.log(calc(net1, [1,1]));
net1 = train(net1, [1,1], [1,2], .1)
console.log(calc(net1, [1,1]));
console.log(net1.fcLayers[0].weights);

console.log('net2:');
console.log(calc(net2, [1,1]));
net2 = train(net2, [1,1], [1,2], .1)
console.log(calc(net2, [1,1]));
console.log(net2.fcLayers[0].weights);
console.log(net2.fcLayers[1].weights);

console.log('net2 longer training:');
for (let i = 0, ie = 100; i < ie; i++)
net2 = train(net2, [1,1], [1,2], .1)
console.log(calc(net2, [1,1]));
console.log(net2.fcLayers[0].weights);
console.log(net2.fcLayers[1].weights);



/*

first layer:
[
    [ 0.34, 0.55 ],
    [ 0.54, 0.15 ],
    [ 0.24, 0.45 ]
]

second layer:
[ 
    [ 0.2, 0.5 ],
    [ 0.3, 0.6 ],
    [ 0.5, 0.2 ]
]

*/


/*
const net = init(2, 4, { neuronsPerHiddenLayer: [ 4 ], actFunc:{outputLayer:EActFunc.LeakyRectifiedLinear} })
        
const DATA = [
    { input: [0,0], output: [0,0,0,1] },
    { input: [0,1], output: [0,1,1,0] },
    { input: [1,0], output: [0,1,1,0] },
    { input: [1,1], output: [1,1,0,1] }
]


    let trainedNet = net
    for (let i = 0; i < 10000; i++) {
        trainedNet = trainSet(trainedNet, DATA, .1)
    }
    DATA.forEach(d => {
        const actual = roundVector(calc(trainedNet, d.input), 1)
        console.log(actual);
        // assert.deepStrictEqual(actual, d.output)
    })
//*/