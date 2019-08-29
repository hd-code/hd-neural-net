import * as fcLayer from "./fc-layers";

interface IHDNeuralNet {
    title?: string
    description?: string
    createdAt: Date
    updatedAt: Date
    layers: ILayers
    learningRate: number
    precision: number
}

interface ILayers {
    fullyConnected: fcLayer.ILayer[]
}