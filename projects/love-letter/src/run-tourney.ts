import { LoveLetterTourney } from "./tourney";
import * as path from "path";
import * as fs from 'fs';
import * as yaml from 'js-yaml';
import yargs from 'yargs';
import { hideBin } from 'yargs/helpers';

interface TourneyConfig {
    bots: Array<{
        name: string;
        command: string;
    }>;
    options: {
        gamesPerMatch: number;
        logDirectory: string;
        reportDirectory: string;
        timeLimitSeconds: number;
        chunkSize: number;
        firstTo: number;
        tourneyName: string;
        debug: {
            tourney: boolean;
            engine: boolean;
        };
    };
}

interface CommandLineArgs {
    config: string;
    gamesPerMatch?: number;
    logDirectory?: string;
    reportDirectory?: string;
    timeLimitSeconds?: number;
    chunkSize?: number;
    firstTo?: number;
    tourneyName?: string;
    debugTourney?: boolean;
    debugEngine?: boolean;
    bots?: string;  // Add bots to command line args
}

// Parse command line arguments with proper typing
const argv = yargs(hideBin(process.argv))
    .options({
        'config': {
            type: 'string',
            default: '../config/tourney.yaml',
            describe: 'Path to config file'
        },
        'gamesPerMatch': { type: 'number', describe: 'Number of games per match' },
        'logDirectory': { type: 'string', describe: 'Directory for logs' },
        'reportDirectory': { type: 'string', describe: 'Directory for reports' },
        'timeLimitSeconds': { type: 'number', describe: 'Time limit per move in seconds' },
        'chunkSize': { type: 'number', describe: 'Number of parallel games' },
        'firstTo': { type: 'number', describe: 'First to N wins' },
        'tourneyName': { type: 'string', describe: 'Tournament name' },
        'debugTourney': { type: 'boolean', describe: 'Enable tournament debug logging' },
        'debugEngine': { type: 'boolean', describe: 'Enable engine debug logging' },
        'bots': { 
            type: 'string', 
            describe: 'Bot configurations in format "name1:command1,name2:command2"'
        }
    })
    .help()
    .parseSync() as CommandLineArgs;

// Load config file
const configPath = path.resolve(__dirname, argv.config);
const configFile = fs.readFileSync(configPath, 'utf8');
const config = yaml.load(configFile) as TourneyConfig;

// Parse bot configurations from command line if provided
let bots = config.bots;
if (argv.bots) {
    bots = argv.bots.split(',').map(botSpec => {
        const [name, command] = botSpec.split(':');
        if (!name || !command) {
            throw new Error(`Invalid bot specification: ${botSpec}. Format should be "name:command"`);
        }
        return { name, command };
    });
}

// Override config with command line arguments
const options = {
    ...config.options,
    ...(argv.gamesPerMatch !== undefined && { gamesPerMatch: argv.gamesPerMatch }),
    ...(argv.logDirectory !== undefined && { logDirectory: argv.logDirectory }),
    ...(argv.reportDirectory !== undefined && { reportDirectory: argv.reportDirectory }),
    ...(argv.timeLimitSeconds !== undefined && { timeLimitSeconds: argv.timeLimitSeconds }),
    ...(argv.chunkSize !== undefined && { chunkSize: argv.chunkSize }),
    ...(argv.firstTo !== undefined && { firstTo: argv.firstTo }),
    ...(argv.tourneyName !== undefined && { tourneyName: argv.tourneyName }),
    debug: {
        tourney: argv.debugTourney ?? config.options.debug?.tourney ?? false,
        engine: argv.debugEngine ?? config.options.debug?.engine ?? false
    }
};

const tourney = new LoveLetterTourney(bots, options);

tourney.run().then(() => {
    console.log("Tournament complete! Check the logs directory for results.");
});
