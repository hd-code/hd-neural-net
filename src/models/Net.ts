import * as FcLayer from './FcLayer';
import * as Vector from './Vector';
import * as Matrix from './Matrix';

// -----------------------------------------------------------------------------

export interface INet {
    createdAt: Date;
    updatedAt: Date;
    fcLayers:  FcLayer.IFcLayer[];
}

// TODO: create Routine!

export function calc(net: INet, input: Vector.TVector): Vector.TVector {
    return net.fcLayers.reduce((result, layer) => FcLayer.calc(layer, result), input);
}

export function train(net: INet, input: Vector.TVector, expOutput: Vector.TVector, learnRate: number): INet {
    let inputs = [input];
    net.fcLayers.forEach((layer,i) => {
        const input = FcLayer.calc(layer,inputs[i]);
        inputs.push(input);
    });

    const lastIndex = net.fcLayers.length - 1;
    let layers: FcLayer.IFcLayer[] = [];

    net.fcLayers.reduceRight( (result,layer,i) => {
        const error = (i === lastIndex)
            ? Vector.sub(inputs[lastIndex+1], expOutput)
            : Matrix.mulVector(result.delta, Matrix.transpose(layer.weights));

        const tmp = FcLayer.train(layer, inputs[i], error, learnRate);
        layers.unshift(tmp.layer);
        return tmp;
    }, {delta: expOutput, layer: net.fcLayers[lastIndex]});

    return {
        createdAt: net.createdAt,
        updatedAt: new Date,
        fcLayers:  layers
    };
}