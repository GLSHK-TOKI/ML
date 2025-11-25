# changeme: application name

## Table of Contents

* [Project Structure](#project-structure)
* [Getting Started](#getting-started)
* [Global Variables](#global-variables)
* [Logging](#logging)
* [Best Practices](#best-practices)
* [Deployment Guide](#deployment-guide)
---

## Project Structure

| Path | Description |
|------|-------------|
| `.env.template` | The template to setup the `.env` file for the local environment. |
| `.eslintrc.js` | The ESLint configuration file for code scan. |
| `index.js` | Entry point of **the cronjob**. |
| `src/` | The directory for most of the application code. |
| `src/configs/` | The configuration files (e.g. database connection, logger configuration, etc.) should be placed here. |
| `src/jobs/` | The **job** for this cronjob pipeline to execute. |
| `src/controllers/` | **Controllers** contain the logic for interacting with models and rendering appropriate views to the client. |
| `src/services/` | **Services** are the reusable modules for the controllers to consume. (e.g. sending emails, interacting with third party APIs, etc.) |
| `test/` | `npm test` will scan this directory for the test cases. |

---

## Getting Started

1. Install the dependencies.

    ```
    $ npm i
    ```

2. Configure the environment variables.
    1. Copy `.env.template` into a new file `.env`
    2. Edit `.env` to setup the local environment.
    3. The application will load the `.env` content as the environment variables on start up.
    4. Note that the `.env` must **NEVER** commit to GIT.
    5. Read more about the behaviour: [dotenv](https://github.com/motdotla/dotenv)

3. Run the test cases.

    ```
    $ npm test
    ```

4. Start the application.

    ```
    $ npm start
    ```

5. Delete the demo codes which are annotated with `FIXME:`.

6. Replace the title of this _README_ file.

7. Replace the `name`, `description` and `author` in _package.json_ and _package-lock.json_.

8. You should update .eslintrc.js for code style your team prefer

9. (Optional) For team with multiple developer, you can enforce better coding standard by enforcing [husky](https://www.npmjs.com/package/husky), it is a pre-commit hooks that run desired steps before commit, and throw error if the step fail

    To enable husky:
    ```
    $ npm run prepare:husky
    ```
    In .env, set:
    ```
    USE_HUSKY=true
    ```
    
    By default, husky will run "npm test" and "npm run lint:fix". You can edit this behavior by updating .husky/pre-commit

10. Enjoy coding!


---


## Logging

Logs should print to the stdout stream. This project uses [pino](https://getpino.io)
as the default logger which prints the logs in JSON format. You could pipe the
output to [pino-pretty](https://github.com/pinojs/pino-pretty) to transform the
logs into human-readable format during local development.

### General Logging

```js
const log = require('./src/services/logger');
log.info('Hello!');

const child = log.child({ ping: 'pong' });
child.info('Bye!');
```

```
# JSON Output
{"level":30,"time":2413987199999,"msg":"Hello!","pid":93901,"hostname":"lab","v":1}
{"level":30,"time":2413987200000,"msg":"Bye!","pid":93901,"hostname":"lab","ping":"pong","v":1}

# Pretty Output
[2413987199999] INFO  (93901 on lab): Hello!
[2413987200000] INFO  (93901 on lab): Bye!
    ping: "pong"
```
Pretty print should only be enabled in local

## Best Practices

* [Cloud for the Community](https://bit.ly/confluence-cloud-for-the-community)
* [Microservice Quality Checklist](https://bit.ly/confluence-microservice-quality-checklist)

---

## Deployment Guide

### Development Environment

1. Environment-dependent configuration should be configured through environment variables.

2. Entry point must be `index.js` (`npm start` must start index.js), due to code injection of analytical tools like AppDynamics.

3. Push to branch `env-d0` to deploy the codes to _d0_ environment.

4. Pipeline flow:

    1. Build stage:

        ```
        $ npm i
        $ npm run build (if "build" step is defined)
        $ npm run test (if "test" step is defined)
        ```

    2. Deploy stage:

        ```
        $ npm start
        ```

5. Visit Kibana to view the logs: https://bit.ly/confluence-kibana-getting-started

### Testing Environment

https://bit.ly/confluence-cicd-deploy-testing

### Production Environment

https://bit.ly/confluence-cicd-deploy-production

---
