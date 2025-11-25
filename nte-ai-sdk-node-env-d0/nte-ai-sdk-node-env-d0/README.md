# Template for Node.js Module

## Getting Started

#### 1. Edit `package.json`

1. Replace `pvt-name` with your PVT name.
2. Replace `module-name` with the module name. It should be unique across your
    team.
3. Add the `description`.

#### 2. Start development

1. Install the dependencies: `npm i`
2. `index.js` is the default entry point for the module. You should replace the
    example with your own code.
3. `test/index.js` contains an example test case. You should replace it with
    yours.

#### 3. Prepare to publish the module

1. Run the test cases (`npm run test`).
2. Update `package.json` > `version`. Publishing the same version more than once
    would cause **unexpected failure** when the others install the module.
3. Update `README.md` (this file) to guide the others to use the module.
