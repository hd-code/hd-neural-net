import { isNumber } from "./helper";

/* --------------------------------- Public --------------------------------- */

export enum EActFunc { 
    Linear,
    Sigmoid,
    Tanh,
    HardTanh,
    RectifiedLinear,
    LeakyRectifiedLinear,
    SoftPlus,
    Softmax,
    Binary 
}

export function isActivationFunction(af: EActFunc): af is EActFunc {
    return isNumber(af) && !!FUNCTIONS[af]
}

export function applyToVector(vector: number[], actFunc: EActFunc, derivative?: boolean): number[] {
    return actFunc !== EActFunc.Softmax
        ? vector.map(value => applyToSingleVal(value, actFunc, derivative))
        : applySoftmaxToVector(vector, derivative)
}

/* --------------------------------- Intern --------------------------------- */

interface IFunction {
    function:   (x: number) => number
    derivative: (x: number) => number
}

const FUNCTIONS :{[func in EActFunc]: IFunction} = {
    [EActFunc.Linear]: {
        function:   (x:number):number => { return x },
        derivative: (x:number):number => { return 1 }
    },
    [EActFunc.Sigmoid]: {
        function:   (x:number):number => { return 1 / (1 + Math.exp(-x)) },
        derivative: (x:number):number => {
            let sig = FUNCTIONS[EActFunc.Sigmoid].function
            return sig(x) * (1 - sig(x))
        }
    },
    [EActFunc.Tanh]: {
        function:   (x:number):number => {
            return (Math.exp(x) - Math.exp(-x)) / (Math.exp(x) + Math.exp(-x))
        },
        derivative: (x:number):number => {
            let tanh = FUNCTIONS[EActFunc.Tanh].function
            return 1 - tanh(x) ** 2
        }
    },
    [EActFunc.HardTanh]: {
        function:   (x:number):number => { return Math.max(-1, Math.min(x, 1)) },
        derivative: (x:number):number => { return (-1 < x && x < 1) ? 1 : 0 }
    },
    [EActFunc.RectifiedLinear]: {
        function:   (x:number):number => { return Math.max(0,x) },
        derivative: (x:number):number => { return x > 0 ? 1 : 0 }
    },
    [EActFunc.LeakyRectifiedLinear]: {
        function:   (x:number):number => { return Math.max(0.01 * x, x) },
        derivative: (x:number):number => { return x > 0 ? 1 : 0.01 }
    },
    [EActFunc.SoftPlus]: {
        function:   (x:number):number => { return Math.log(1 + Math.exp(x)) },
        derivative: (x:number):number => {
            const sigmoid = FUNCTIONS[EActFunc.Sigmoid].function
            return sigmoid(x)
        }
    },
    // TODO!
    [EActFunc.Softmax]: {
        function:   (x:number):number => { return x },
        derivative: (x:number):number => { return 1 }
    },
    [EActFunc.Binary]: {
        function:   (x:number):number => { return x < 0 ? 0 : 1 },
        derivative: (x:number):number => { return x === 0 ? 0 : Math.random() }
    }
}

function applyToSingleVal(value: number, func: EActFunc, derivative?:boolean): number {
    return !derivative
        ? FUNCTIONS[func].function(value)
        : FUNCTIONS[func].derivative(value)
}

function applySoftmaxToVector(values: number[], derivative?: boolean): number[] {
    const divider = values.reduce((result, value) => result + Math.exp(value), 0)
    return !derivative
        ? values.map(value => Math.exp(value) / divider)
        : values.map(value => 1) // TODO!!! 
}