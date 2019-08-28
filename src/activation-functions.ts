/*********************************** Public ***********************************/

export enum EActFunction { SIGMOID, RELU, TANH, LINEAR, BINARY }

export function apply(value :number, func :EActFunction, derivative?:boolean) :number {
    return !derivative
        ? ACTIVATION_FUNCTIONS[func].function(value)
        : ACTIVATION_FUNCTIONS[func].derivative(value)
}

export function applyToVector(vector :number[], actFunc :EActFunction, derivative?:boolean) :number[] {
    return vector.map(value => {
        return apply(value, actFunc, derivative)
    })
}

/*********************************** Intern ***********************************/

interface IFunction {
    function:   (x: number) => number
    derivative: (x: number) => number
}

const ACTIVATION_FUNCTIONS:IFunction[] = [
    { // SIGMOID
        function:   (x:number):number => {return 1 / (1 + Math.exp(-x))},
        derivative: (x:number):number => {
            let sig = ACTIVATION_FUNCTIONS[EActFunction.SIGMOID].function
            return sig(x) * (1 - sig(x))
        }
    },
    { // RELU
        function:   (x:number):number => {return Math.max(0,x)},
        derivative: (x:number):number => {return x > 0 ? 1 : 0}
    },
    { // TANH
        function:   (x:number):number => {return (Math.exp(x) - Math.exp(-x)) / (Math.exp(x) + Math.exp(-x))},
        derivative: (x:number):number => {return 1 - ACTIVATION_FUNCTIONS[EActFunction.TANH].function(x) ** 2}
    },
    { // LINEAR
        function:   (x:number):number => {return x},
        derivative: (x:number):number => {return 1}
    },
    { // BINARY
        function:   (x:number):number => {return x < 0 ? 0 : 1},
        derivative: (x:number):number => {return x === 0 ? 0 : Math.random()}
    },
]