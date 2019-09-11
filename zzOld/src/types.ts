export interface IHDNeuralNet {
    title?: string
    description?: string
    createdAt: Date
    updatedAt: Date
    learningRate: number
    precision: number
    layers: IFCLayer[]
}

export interface INetOptions {
    title?: string
    description?: string

    learningRate?: number
    precision?: number
    noBias?: boolean

    activationFunctions?: {
        allLayers?: EActFunction
        allHiddenLayers?: EActFunction
        hiddenLayers?: EActFunction[]
        outputLayer?: EActFunction
    }
}

export interface ITrainingData {
    input: number[]
    output: number[]
}

export enum EActFunction { SIGMOID, RELU, TANH, LINEAR, BINARY }

export interface IFCLayer {
    actFunction: EActFunction
    weights: number[][] // [neurons on this layer][weights for outputs from prev layer]
}

export interface IFCLayerConfig {
    actFunction: EActFunction
    numOfNeurons: number
}