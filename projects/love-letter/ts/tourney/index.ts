import { LoveLetterEngine } from "../engine";
import * as fs from 'fs';
import * as path from 'path';
import { ChildProcess, spawn } from 'child_process';
import { Readable, Writable } from 'stream';

interface BotSpec {
  name: string;
  command: string;  // path to the bot executable/script
  elo?: number;     // track ELO rating
}


interface BotProcess {
    process: ChildProcess;
    stdin: Writable;
    stdout: Readable;
    name: string;
}
  

interface MatchResult {
  bot1: string;
  bot2: string;
  bot1WinsFirst: number;
  bot1WinsSecond: number;
  gamesPlayedFirst: number;
  gamesPlayedSecond: number;
}

export interface TourneyOptions {
  gamesPerMatch?: number;
  logDirectory?: string;
  reportDirectory?: string;
  timeLimitSeconds?: number;
  tourneyName?: string;
  chunkSize?: number;
  firstTo?: number;
  debug?: {
    tourney: boolean;
    engine: boolean;
  };
}

const DEFAULT_ELO = 1500;
const K_FACTOR = 32;

function generateRandomTourneyName(): string {
  const words = ["Mountain", "Ocean", "Forest", "River", "Desert", "Valley", "Storm", "Thunder", "Lightning", "Wind", "Rain", "Cloud", "Star", "Moon", "Sun", "Dawn", "Dusk", "Shadow", "Light", "Fire", "Ice", "Earth", "Sky", "Wave", "Crystal", "Dragon", "Phoenix", "Eagle", "Wolf", "Lion", "Tiger", "Bear", "Hawk", "Falcon", "Raven", "Owl", "Serpent", "Dolphin", "Whale"];
  
  const chosenWords = Array.from({length: 3}, () => 
    words[Math.floor(Math.random() * words.length)]
  );
  
  return chosenWords.join("");
}

function calculateEloChange(winnerElo: number, loserElo: number): number {
    const expectedScore = 1 / (1 + Math.pow(10, (loserElo - winnerElo) / 400));
    return K_FACTOR * (1 - expectedScore);
}

async function getBotMove(currentPlayer: number, botProcess: BotProcess, gameLog: string[], timeLimit: number): Promise<string> {
    return new Promise((resolve, reject) => {        
        let output = '';
        const timer = setTimeout(() => {
            reject('timeout');
        }, timeLimit * 1000);

        const moveHandler = (data: Buffer) => {
            output += data.toString();
            if (output.includes('\n')) {
                // console.log("Output:", output);
                if (botProcess.process.stdout) {
                    botProcess.process.stdout.removeListener('data', moveHandler);
                }
                clearTimeout(timer);
                resolve(output.trim());
            }
        };

        botProcess.process.stdout!.on('data', moveHandler);

        // Send game state to bot
        const otherPlayer = (currentPlayer + 1) % 2;
        const filteredLog = gameLog.filter(line => !line.includes(`p${otherPlayer + 1}|hidden`));
        botProcess.stdin.write(`move ${timeLimit}\n${filteredLog.join('\n')}\n\n`);
    });
}

export class LoveLetterTourney {
  private bots: BotSpec[];
  private options: TourneyOptions;
  private results: Map<string, MatchResult>;
  private eloRatings: Map<string, number>;
  private debugTourney: boolean;
  private debugEngine: boolean;

  private log(message: string) {
    if (this.debugTourney) {
      console.log(`[DEBUG][Tourney] ${message}`);
    }
  }

  constructor(bots: BotSpec[], options?: TourneyOptions) {
    this.bots = bots;
    this.options = {
      gamesPerMatch: 100,
      logDirectory: "./logs",
      reportDirectory: "./reports",
      timeLimitSeconds: 1,
      tourneyName: generateRandomTourneyName(),
      debug: {
        tourney: false,
        engine: false
      },
      ...options
    };
    this.debugTourney = this.options.debug?.tourney || false;
    this.debugEngine = this.options.debug?.engine || false;
    this.results = new Map();
    this.eloRatings = new Map(bots.map(bot => [bot.name, DEFAULT_ELO]));
  }

    private getMatchKey(p1: string, p2: string): string {
      // Always use alphabetical order to ensure consistent key regardless of who goes first
      return [p1, p2].sort().join(':');
    }
  

    private updateResults(p1: string, p2: string, bot1WentFirst: boolean, p1Won: boolean) {
        const key = this.getMatchKey(p1, p2);
        
        const bot1 = (bot1WentFirst ? p1 : p2);
        const bot2 = bot1WentFirst ? p2 : p1;
        
        let result = this.results.get(key) || {
            bot1,
            bot2,
            bot1WinsFirst: 0,
            bot1WinsSecond: 0,
            gamesPlayedFirst: 0,
            gamesPlayedSecond: 0
        };
        const bot1Won = bot1WentFirst ? p1Won : !p1Won;

        if (bot1WentFirst) {
            result.gamesPlayedFirst++;
            if (bot1Won) result.bot1WinsFirst++;
        } else {
            result.gamesPlayedSecond++;
            if (bot1Won) result.bot1WinsSecond++;
        }

        this.results.set(key, result);

        // Update ELO
        const winner = p1Won ? p1 : p2;
        const loser = p1Won ? p2 : p1;
        
        const eloChange = calculateEloChange(
            this.eloRatings.get(winner)!,
            this.eloRatings.get(loser)!
        ); 
        
        this.eloRatings.set(winner, this.eloRatings.get(winner)! + eloChange);
        this.eloRatings.set(loser, this.eloRatings.get(loser)! - eloChange);
    }    private spawnBot(botSpec: BotSpec): BotProcess {
        const parts = botSpec.command.split(' ');
        const process = spawn(parts[0], parts.slice(1), {
            stdio: ['pipe', 'pipe', 'pipe']
        });
        
        const botProcess = {
            process,
            stdin: process.stdin,
            stdout: process.stdout,
            name: botSpec.name
        };
    
        this.setupBotErrorHandling(botProcess);
        
        return botProcess;
    }

    private setupBotErrorHandling(botProcess: BotProcess) {
        botProcess.process.on('error', (error) => {
            console.error(`Bot ${botProcess.name} process error:`, error);
        });
    
        botProcess.process.stderr!.on('data', (data) => {
            console.error(`Bot ${botProcess.name} stderr:`, data.toString());
        });
    }

    private async playOneGame(p1: BotSpec, p2: BotSpec, gameNumber: string, logDir: string, firstTo = 10): Promise<void> {
        this.log(`Starting game ${gameNumber} between ${p1.name} and ${p2.name}`);

        const bot1Process = this.spawnBot(p1);
        const bot2Process = this.spawnBot(p2);
        const botProcesses = [bot1Process, bot2Process];

        try {
            let subGameNumber = 0;
            let w1 = 0;
            let w2 = 0;
            while (w1 < firstTo && w2 < firstTo) {
                this.log(`Starting subgame ${subGameNumber} (Score: ${w1}-${w2})`);
                const logLines: string[] = [];
                const slot1 = (subGameNumber % 2 === 0 ? p1 : p2);
                const slot2 = (subGameNumber % 2 === 0 ? p2 : p1);
                const engine = new LoveLetterEngine(2, `${gameNumber}-${subGameNumber}`, {
                    logCallback: (line) => logLines.push(line),
                    debug: this.debugEngine
                });
                engine.startGameLog([slot1.name, slot2.name]);

                while (!engine.isGameOver()) {
                    const currentPlayer = engine.getCurrentPlayerIndex();
                    engine.drawCardForCurrentPlayer();

                    try {
                        const move = await getBotMove(
                            currentPlayer,
                            botProcesses[(currentPlayer + subGameNumber) % 2],
                            logLines,
                            this.options.timeLimitSeconds!
                        );

                        const [card, target] = move.split(/[\s|]/).map(x => parseInt(x));
                        engine.makeMove(currentPlayer, card, target);

                    } catch (error) {
                        if (error === 'timeout') {
                            this.log(`Player ${currentPlayer + 1} (${botProcesses[(currentPlayer + subGameNumber) % 2].name}) lost due to timeout`);
                            engine.log(`|lose|p${currentPlayer + 1}|timeout`);
                        } else {
                            this.log(`Player ${currentPlayer + 1} (${botProcesses[(currentPlayer + subGameNumber) % 2].name}) lost due to invalid move`);
                            engine.log(`|lose|p${currentPlayer + 1}|invalid`);
                        }
                        engine.log(`|end|p${(currentPlayer + 1) % 2 + 1}|win`);
                        break;
                    }
                }

                // Write game log
                fs.writeFileSync(
                    path.join(logDir, `${gameNumber}-${subGameNumber}.log`),
                    logLines.join('\n'),
                    'utf8'
                );
                
                // Update results - check if this was a "bot1 first" game
                const winner = logLines[logLines.length - 1].split('|')[2];
                if (winner === 'p1') {
                    if (subGameNumber % 2 === 0) {
                        w1++;
                        this.log(`${slot1.name} won subgame ${subGameNumber} as Player 1 (New score: ${w1}-${w2})`);
                    } else {
                        w2++;
                        this.log(`${slot2.name} won subgame ${subGameNumber} as Player 1 (New score: ${w1}-${w2})`);
                    }
                } else {
                    if (subGameNumber % 2 === 0) {
                        w2++;
                        this.log(`${slot2.name} won subgame ${subGameNumber} as Player 2 (New score: ${w1}-${w2})`);
                    } else {
                        w1++;
                        this.log(`${slot1.name} won subgame ${subGameNumber} as Player 2 (New score: ${w1}-${w2})`);
                    }
                }
            
                subGameNumber++;
            }
            const p1Won = w1 === firstTo;
            const isBot1First = gameNumber.endsWith('Afirst');
            this.updateResults(p1.name, p2.name, isBot1First, p1Won);
            
            this.log(`Game ${gameNumber} finished: ${p1Won ? p1.name : p2.name} won (Final score: ${w1}-${w2})`);

        } finally {
            bot1Process.process.kill();
            bot2Process.process.kill();
        }
    }
    private generateReport(): string {
        let report = `Tournament Report: ${this.options.tourneyName}\n`;
        report += '=====================================\n\n';

        // Pairwise results
        report += 'Pairwise Results:\n';
        for (const result of this.results.values()) {
            report += `\n${result.bot1} vs ${result.bot2}:\n`;
            report += `  As First Player: ${result.bot1WinsFirst}/${result.gamesPlayedFirst} (${(result.bot1WinsFirst/result.gamesPlayedFirst*100).toFixed(1)}%)\n`;
            report += `  As Second Player: ${result.bot1WinsSecond}/${result.gamesPlayedSecond} (${(result.bot1WinsSecond/result.gamesPlayedSecond*100).toFixed(1)}%)\n`;
            report += `  Total: ${result.bot1WinsFirst + result.bot1WinsSecond}/${result.gamesPlayedFirst + result.gamesPlayedSecond} (${((result.bot1WinsFirst + result.bot1WinsSecond)/(result.gamesPlayedFirst + result.gamesPlayedSecond)*100).toFixed(1)}%)\n`;
        }

        // ELO Rankings
        report += '\nFinal ELO Rankings:\n';
        const sortedElo = Array.from(this.eloRatings.entries())
            .sort(([,a], [,b]) => b - a);
        
        sortedElo.forEach(([name, elo], index) => {
            report += `${index + 1}. ${name}: ${Math.round(elo)}\n`;
        });

        return report;
    }

    // Add this new helper function
    private chunkGames(games: Array<[BotSpec, BotSpec, string]>, chunks: number = 8): Array<Array<[BotSpec, BotSpec, string]>> {
      const result = [];
      const chunkSize = Math.ceil(games.length / chunks);
      for (let i = 0; i < games.length; i += chunkSize) {
          result.push(games.slice(i, i + chunkSize));
      }
      return result;
    }

    // Modify the run method
    public async run() {
      const logDir = this.options.logDirectory!;
      const reportDir = this.options.reportDirectory!;
      if (!fs.existsSync(logDir)) {
          fs.mkdirSync(logDir, { recursive: true });
      }
      if (!fs.existsSync(reportDir)) {
          fs.mkdirSync(reportDir, { recursive: true });
      }

      // Generate all game combinations first
      const allGames: Array<[BotSpec, BotSpec, string]> = [];
      for (let i = 0; i < this.bots.length; i++) {
          for (let j = i + 1; j < this.bots.length; j++) {
              for (let g = 0; g < this.options.gamesPerMatch!; g++) {
                  allGames.push([
                      this.bots[i],
                      this.bots[j],
                      `${this.options.tourneyName}-Game-${i}-${j}-${g}-Afirst`
                  ]);
                  allGames.push([
                      this.bots[j],
                      this.bots[i],
                      `${this.options.tourneyName}-Game-${i}-${j}-${g}-Bfirst`
                  ]);
              }
          }
      }

      // Split games into chunks and run them in parallel
      const chunks = this.chunkGames(allGames, this.options.chunkSize);
      await Promise.all(
          chunks.map(async (chunk) => {
              for (const [bot1, bot2, gameNumber] of chunk) {
                  await this.playOneGame(bot1, bot2, gameNumber, logDir, this.options.firstTo);
              }
          })
      );

      // Generate and save report
      const reportPath = path.join(reportDir, `${this.options.tourneyName}-report.log`);
      fs.writeFileSync(reportPath, this.generateReport(), 'utf8');
    }
}
