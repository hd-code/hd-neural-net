const { init, calcSet, trainSet } = require('../build/main');
const { deepClone } = require('../build/helper');

const NET = init(2, 4, {
    neuronsPerHiddenLayer: [16]
})

const DATA = [
    { input: [0,0], output: [0,0,0,1] },
    { input: [0,1], output: [0,1,1,0] },
    { input: [1,0], output: [0,1,1,0] },
    { input: [1,1], output: [1,1,0,1] }
]

console.log('Before:', calcSet(NET, DATA));

let newNet = deepClone(NET)
for (let i = 0; i < 10000; i++) {
    newNet = trainSet(newNet, DATA)
}

console.log('After:', calcSet(newNet, DATA));