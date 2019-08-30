import { IHDNeuralNet, ITraingData, train, calc } from "./index";
import { init } from "./fc-layers";
import { EActFunction } from "./activation-functions";

let net :IHDNeuralNet = {
    createdAt: new Date,
    updatedAt: new Date,
    learningRate: .01,
    precision: .1,
    layers: init(2, {numOfNeurons: 2, actFunction: EActFunction.BINARY}, [{numOfNeurons: 2}])
}

let data :ITraingData = {
    input: [1,1],
    output:[1,0]
}

let tmp = train(data, net, .001, .1)

if (tmp === null) {
    console.log('calculation failed')
} else {
    console.log(calc(data.input, tmp));
}