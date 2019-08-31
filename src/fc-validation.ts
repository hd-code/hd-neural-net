import { isArrayOf, isMatrix } from "./helper";
import { ILayer } from "./fc-layers";
import { isActivationFunction } from "./activation-functions";

/* --------------------------------- Public --------------------------------- */

export function isFCNet(layers: ILayer[]): layers is ILayer[] {
    let result = true

    // type checking
    if (!isArrayOf(layers, isLayer)) {
        console.error('layers have wrong structure or datatypes')
        return false
    }

    // check if all layer outputs match next layers inputs
    let indexOfOutputLayer = layers.length - 1
    let numOfInAndOutputs = layers.map(layer => getNumOfInAndOutputs(layer))
    numOfInAndOutputs.reduce((outputs, layer, i) => {
        if (outputs > layer.inputs) {
            result = false
            let error = ''
            switch (i) {
                case 0:
                    error = 'There are too many input values for this network'
                    break;
                case indexOfOutputLayer:
                    error = 'Last hidden layer has too many outputs for the output layer'
                    break;
                default:
                    error = i + '. hidden layer has too many outputs for the following layer'
            }
            console.error(error)
        }
        return layer.outputs
    }, numOfInAndOutputs[0].inputs)

    return result
}

export function getNumOfNetInAndOutputs(layers: ILayer[]): {inputs:number, outputs:number} {
    return {
        inputs: getNumOfInAndOutputs(layers[0]).inputs,
        outputs: getNumOfInAndOutputs(layers[layers.length - 1]).outputs,
    }
}

/* --------------------------------- Intern --------------------------------- */

function isLayer(layer: ILayer): layer is ILayer {
    return 'actFunction' in layer && isActivationFunction(layer.actFunction)
        && 'weights' in layer  && isMatrix(layer.weights)
}

function getNumOfInAndOutputs(layer: ILayer): {inputs:number, outputs:number} {
    return {
        inputs: layer.weights[0].length,
        outputs: layer.weights.length
    }
}