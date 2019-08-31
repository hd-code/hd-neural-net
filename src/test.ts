import { IHDNeuralNet, ITraingData, train, calc, init, IOptions } from "./index";
import { EActFunction } from "./activation-functions";

let net = init(2, 2, [10])
let data: ITraingData[] = [
    { input: [0,0], output:[0,0] },
    { input: [0,1], output:[0,1] },
    { input: [1,0], output:[0,1] },
    { input: [1,1], output:[1,0] },
]
let input = data.map(d => d.input)

if (net) {
    console.log('Start training');
    
    let tmp = train(data, net, .01, .01)

    if (tmp === null) {
        console.log('calculation failed')
    } else {
        console.log(calc(input, tmp));
    }
}





// let options :IOptions = {
//     activationFunctions: {
//         outputLayer: EActFunction.RELU
//     }
// }
// let net = init(1, 4, [8], options)
// let data: ITraingData[] = [
//     { input: [0], output:[0,0,0,0] },
//     { input: [1], output:[0,0,0,1] },
//     { input: [2], output:[0,0,1,0] },
//     { input: [3], output:[0,0,1,1] },
//     { input: [4], output:[0,1,0,0] },
//     { input: [5], output:[0,1,0,1] },
//     { input: [6], output:[0,1,1,0] },
//     { input: [7], output:[0,1,1,1] },
//     { input: [8], output:[1,0,0,0] },
// ]
// let input = data.map(d => d.input)

// if (net) {
//     console.log('Start training');
    
//     let tmp = train(data, net, .01, .1)

//     if (tmp === null) {
//         console.log('calculation failed')
//     } else {
//         console.log(calc(input, tmp));
//         console.log(calc([9], tmp));
//     }
// }