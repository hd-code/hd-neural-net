import * as fcLayer from "./fc-layers";

interface hdNeuralNet {
    title?: string
    description?: string
    createdAt: Date
    updatedAt: Date
    layers: ILayers
    learningRate: number
    precision: number
}

interface ILayers {
    fullConnected: fcLayer.ILayer[]
}