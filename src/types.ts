import { EActFunction } from "./activation-functions";

export interface INet extends Array<IFCLayer> {}

export interface IFCLayer {
    actFunction: EActFunction
    weights: number[][] // [neurons on prev layer][neurons on this layer]
}

export interface IFCLayerConfig {
    actFunction?: EActFunction
    numOfNeurons: number
}