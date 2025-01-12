import { LoveLetterTourney } from "./tourney";
import * as path from "path";

const bots = [
    {
        name: "RandomBot1",
        command: "python3.10 -m bots.random_bot"
    },
    {
        name: "RandomBot2",
        command: "python3.10 -m bots.random_bot"
    },
];

const tourney = new LoveLetterTourney(bots, {
    gamesPerMatch: 125000, // n * (n - 1) times more than this
    logDirectory: path.join(__dirname, "../.logs/bad-logs"),
    reportDirectory: path.join(__dirname, "../.reports"),
    timeLimitSeconds: 60,
    chunkSize: 8,
    firstTo: 1
});

tourney.run().then(() => {
    console.log("Tournament complete! Check the logs directory for results.");
});
