import { isNumber } from "../lib/hd-helper";

// -----------------------------------------------------------------------------

export enum EActFunction { 
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

export function isActFunction(af: EActFunction): af is EActFunction {
    return isNumber(af) && FUNCTIONS[af] !== undefined;
}

export function applyToVector(vector: number[], actFunc: EActFunction, derivative?: boolean): number[] {
    return actFunc !== EActFunction.Softmax
        ? vector.map(value => applyToSingleVal(value, actFunc, derivative))
        : applySoftmaxToVector(vector, derivative);
}

// -----------------------------------------------------------------------------

interface IFunction {
    function:   (x: number) => number;
    derivative: (x: number) => number;
}

const FUNCTIONS :{[func in EActFunction]: IFunction} = {
    [EActFunction.Linear]: {
        function:   (x:number):number => { return x; },
        derivative: (x:number):number => { return 1; }
    },
    [EActFunction.Sigmoid]: {
        function:   (x:number):number => { return 1 / (1 + Math.exp(-x)); },
        derivative: (x:number):number => {
            const sig = FUNCTIONS[EActFunction.Sigmoid].function;
            return sig(x) * (1 - sig(x));
        }
    },
    [EActFunction.Tanh]: {
        function:   (x:number):number => {
            return (Math.exp(x) - Math.exp(-x)) / (Math.exp(x) + Math.exp(-x));
        },
        derivative: (x:number):number => {
            const tanh = FUNCTIONS[EActFunction.Tanh].function;
            return 1 - tanh(x) ** 2;
        }
    },
    [EActFunction.HardTanh]: {
        function:   (x:number):number => { return Math.max(-1, Math.min(x, 1)); },
        derivative: (x:number):number => { return (-1 < x && x < 1) ? 1 : 0; }
    },
    [EActFunction.RectifiedLinear]: {
        function:   (x:number):number => { return Math.max(0,x); },
        derivative: (x:number):number => { return x > 0 ? 1 : 0; }
    },
    [EActFunction.LeakyRectifiedLinear]: {
        function:   (x:number):number => { return Math.max(0.01 * x, x); },
        derivative: (x:number):number => { return x > 0 ? 1 : 0.01; }
    },
    [EActFunction.SoftPlus]: {
        function:   (x:number):number => { return Math.log(1 + Math.exp(x)); },
        derivative: (x:number):number => {
            const sigmoid = FUNCTIONS[EActFunction.Sigmoid].function;
            return sigmoid(x);
        }
    },
    // TODO!
    [EActFunction.Softmax]: {
        function:   (x:number):number => { return x; },
        derivative: (x:number):number => { return 1; }
    },
    [EActFunction.Binary]: {
        function:   (x:number):number => { return x < 0 ? 0 : 1; },
        derivative: (x:number):number => { return x === 0 ? 0 : Math.random(); }
    }
};

function applyToSingleVal(value: number, func: EActFunction, derivative?:boolean): number {
    return !derivative
        ? FUNCTIONS[func].function(value)
        : FUNCTIONS[func].derivative(value);
}

function applySoftmaxToVector(values: number[], derivative?: boolean): number[] {
    const divider = values.reduce((result, value) => result + Math.exp(value), 0);
    return !derivative
        ? values.map(value => Math.exp(value) / divider)
        : values.map(value => 1); // TODO!!! 
}