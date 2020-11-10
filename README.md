# HD Neural Net

This is my playground for developing a neural net library in TypeScript.

I tune in from time to time to checkout some new ideas about the whole topic.

## Installation

```sh
# clone repo
git clone https://github.com/hd-code/hd-neural-net.git

# go to project directory
cd hd-neural-net

# install dependencies
npm install
```

## Usage

... (I have not decided on the final high-level API yet)

## Development

Project structure:

- `dist/` holds the compiled and minified code for distribution. This is generated automatically during a build process.
- `src/` holds all the source code, which will be compiled to `dist/`
- `test/` holds automatic test scripts for the code in `src/`

Helpful commands:

- `npm run build` will compile and minify the project to `dist/`
- `npm run lint` will check and fix to some degree the syntax of the TypeScript files in `src/` and `test/`
- `npm test` will run all tests in `test/` and log the results