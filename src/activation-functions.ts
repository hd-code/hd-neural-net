import { isNumber } from "./helper";

/* --------------------------------- Public --------------------------------- */

export enum ACT_FUNC { 
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

export function isActivationFunction(af: ACT_FUNC): af is ACT_FUNC {
    return isNumber(af) && !!ACTIVATION_FUNCTIONS[af]
}

export function applyToVector(vector: number[], actFunc: ACT_FUNC, 
    derivative?: boolean): number[]
{
    return actFunc !== ACT_FUNC.Softmax
        ? vector.map(value => applyToSingleVal(value, actFunc, derivative))
        : applySoftmaxToVector(vector, derivative)
}

/* --------------------------------- Intern --------------------------------- */

interface IFunction {
    function:   (x: number) => number
    derivative: (x: number) => number
}

const ACTIVATION_FUNCTIONS :{[func in ACT_FUNC]: IFunction} = {
    [ACT_FUNC.Linear]: {
        function:   (x:number):number => { return x },
        derivative: (x:number):number => { return 1 }
    },
    [ACT_FUNC.Sigmoid]: {
        function:   (x:number):number => { return 1 / (1 + Math.exp(-x)) },
        derivative: (x:number):number => {
            let sig = ACTIVATION_FUNCTIONS[ACT_FUNC.Sigmoid].function
            return sig(x) * (1 - sig(x))
        }
    },
    [ACT_FUNC.Tanh]: {
        function:   (x:number):number => {
            return (Math.exp(x) - Math.exp(-x)) / (Math.exp(x) + Math.exp(-x))
        },
        derivative: (x:number):number => {
            let tanh = ACTIVATION_FUNCTIONS[ACT_FUNC.Tanh].function
            return 1 - tanh(x) ** 2
        }
    },
    [ACT_FUNC.HardTanh]: {
        function:   (x:number):number => { return Math.max(-1, Math.min(x, 1)) },
        derivative: (x:number):number => { return (-1 < x && x < 1) ? 1 : 0 }
    },
    [ACT_FUNC.RectifiedLinear]: {
        function:   (x:number):number => { return Math.max(0,x) },
        derivative: (x:number):number => { return x > 0 ? 1 : 0 }
    },
    [ACT_FUNC.LeakyRectifiedLinear]: {
        function:   (x:number):number => { return Math.max(0.01 * x, x) },
        derivative: (x:number):number => { return x > 0 ? 1 : 0.01 }
    },
    [ACT_FUNC.SoftPlus]: {
        function:   (x:number):number => { return Math.log(1 + Math.exp(x)) },
        derivative: (x:number):number => {
            const sigmoid = ACTIVATION_FUNCTIONS[ACT_FUNC.Sigmoid].function
            return sigmoid(x)
        }
    },
    [ACT_FUNC.Softmax]: {
        function:   (x:number):number => { return x },
        derivative: (x:number):number => { return 1 }
    },
    [ACT_FUNC.Binary]: {
        function:   (x:number):number => { return x < 0 ? 0 : 1 },
        derivative: (x:number):number => { return x === 0 ? 0 : Math.random() }
    }
}

function applySoftmaxToVector(values: number[], derivative?: boolean): number[] {
    const divider = values.reduce((result, value) => result + Math.exp(value), 0)
    return !derivative
        ? values.map(value => Math.exp(value) / divider)
        : values.map(value => 1) // TODO!!! 
}

function applyToSingleVal(value: number, func: ACT_FUNC, derivative?:boolean): number {
    return !derivative
        ? ACTIVATION_FUNCTIONS[func].function(value)
       :  ACTIVATION_FUNCTIONS[func].derivative(value)
}