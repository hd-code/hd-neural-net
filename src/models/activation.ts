import { isInteger } from "../helper/type-guards";

export enum Activation {
    Linear,
    Sigmoid,
    Tanh,
    HardTanh,
    RectifiedLinear,
    LeakyRectifiedLinear,
    SoftPlus,
    Softmax, // not fully implemented
    Binary,
}

export function isActivation(activation: any): activation is Activation {
    return isInteger(activation) && Activation.Linear <= activation && activation <= Activation.Binary;
}

export function calcActivation(activation: Activation, input: number[], derivative = false): number[] {
    const func = !derivative ? functions[activation]?.function : functions[activation]?.derivative;
    if (func === undefined) {
        return derivative ? softmaxDerivative(input) : softmax(input);
    }
    return input.map(x => func(x));
}

// -----------------------------------------------------------------------------

function softmax(input: number[]): number[] {
    const sumExp = input.reduce((sum, x) => sum + Math.exp(x), 0);
    return input.map(x => x / sumExp);
}

function softmaxDerivative(input: number[]): number[] {
    return input.map(x => 1); // TODO
}

// -----------------------------------------------------------------------------

interface IFunction {
    function:   (x: number) => number
    derivative: (x: number) => number
}

const functions :{[func in Activation]?: IFunction} = {
    [Activation.Linear]: {
        function:   x => x,
        derivative: _ => 1,
    },

    [Activation.Sigmoid]: {
        function:   x => 1 / (1 + Math.exp(-x)),
        derivative: x => {
            const sig = 1 / (1 + Math.exp(-x));
            return sig * (1 - sig);
        },
    },

    [Activation.Tanh]: {
        function: x => (Math.exp(x) - Math.exp(-x)) / (Math.exp(x) + Math.exp(-x)),
        derivative: (x:number):number => {
            let tanh = (Math.exp(x) - Math.exp(-x)) / (Math.exp(x) + Math.exp(-x));
            return 1 - tanh ** 2;
        },
    },

    [Activation.HardTanh]: {
        function:   x => Math.max(-1, Math.min(x, 1)),
        derivative: x => (-1 < x && x < 1) ? 1 : 0,
    },

    [Activation.RectifiedLinear]: {
        function:   x => Math.max(0,x),
        derivative: x => x > 0 ? 1 : 0,
    },

    [Activation.LeakyRectifiedLinear]: {
        function:   x => Math.max(0.01 * x, x),
        derivative: x => x > 0 ? 1 : 0.01,
    },

    [Activation.SoftPlus]: {
        function:   x => Math.log(1 + Math.exp(x)),
        derivative: x => 1 / (1 + Math.exp(-x))
    },

    [Activation.Binary]: {
        function:   x => x < 0 ? 0 : 1,
        derivative: x => x === 0 ? 0 : Math.random(),
    }
}
