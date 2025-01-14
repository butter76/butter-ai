# BUTTER AI

This is a repo for Butter AI that covers many different 2-player games of varying complexity.

This a dual python + yarn mono-workspace that combines both Typed Python & Typescript.

## How to get started

### Setting up VSCode

The repo includes VSCode settings that will help you get started with Pylance and Typescript debugging support.

### Installing Python

It is a requirement to use Python 3.11 aliased to `python3.11`. Be sure that `pip` is also installed (code will use `python3.11 -m pip`).

### Installing Yarn

It is a requirement to use Yarn 1.22, as specified in the package.json.

```
npm install -g corepack
corepack enable
yarn install
yarn setup
```

Most scripts are run using `yarn`. Check the `package.json` in each project folder for all scripts.