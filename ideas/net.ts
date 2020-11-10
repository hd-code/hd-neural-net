import { isArray } from '../helper/type-guards';
import { calcLayer, isLayer, Layer } from './layer';

// -----------------------------------------------------------------------------

export interface Net {
    layers: Layer[];
}

export function isNet(net: any): net is Net {
    return 'layers' in net && isArray(net.layers, isLayer);
}

export function initNet() {}

export function calcNet(net: Net, input: number[]): number[] {
    return net.layers.reduce((input, layer) => calcLayer(layer, input), input);
}
