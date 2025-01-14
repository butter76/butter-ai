/**
 * Love Letter Engine
 * ------------------
 * Implements the game logic for Love Letter according to the specifications.
 * This includes:
 *   - Deck management
 *   - Turn structure
 *   - Card effects
 *   - Game logging
 */

export interface PlayerState {
  hand: number[];
  protected: boolean; // whether Handmaid (4) effect is active
  eliminated: boolean; // track if player is knocked out
}

export interface GameOptions {
  seed?: string;  // optional random seed
  logCallback?: (line: string) => void;
  debug?: boolean;  // Add debug flag
}

export class LoveLetterEngine {
  private deck: number[];
  private players: PlayerState[];
  private currentPlayerIndex: number;
  private logCallback: (line: string) => void;
  private gameId: string;
  private removedCard: number | null;
  private gameEnded: boolean;
  private debug: boolean;

  constructor(private numPlayers: number = 2, gameId: string, options?: GameOptions) {
    // Initialize random deck of cards (16 total).
    // Cards: 1(Guard)x5, 2(Priest)x2, 3(Baron)x2, 4(Handmaid)x2,
    // 5(Prince)x2, 6(King)x1, 7(Countess)x1, 8(Princess)x1
    // Simple unseeded shuffle for demonstration. For reproducible
    // results, consider using a seeded RNG if options.seed is provided.
    this.deck = this.generateDeck();
    this.shuffle(this.deck);

    // Remove top card (face-down)
    this.removedCard = this.deck.shift() || null;

    // Initialize each player's state
    this.players = [];
    for (let i = 0; i < numPlayers; i++) {
      const card = this.deck.shift();
      this.players.push({
        hand: card !== undefined ? [card] : [],
        protected: false,
        eliminated: false,
      });
    }

    // Set up logging
    this.logCallback = options?.logCallback || (() => {});
    this.gameId = gameId;
    this.currentPlayerIndex = 0;
    this.gameEnded = false;
    this.debug = options?.debug || false;
  }

  private generateDeck(): number[] {
    return [
      1,1,1,1,1,   // Guards (5)
      2,2,         // Priests (2)
      3,3,         // Barons (2)
      4,4,         // Handmaids (2)
      5,5,         // Princes (2)
      6,           // King (1)
      7,           // Countess (1)
      8            // Princess (1)
    ];
  }

  private shuffle(array: number[]) {
    for (let i = array.length - 1; i > 0; i--) {
      const j = Math.floor(Math.random() * (i + 1));
      [array[i], array[j]] = [array[j], array[i]];
    }
  }

  public startGameLog(playerNames: string[]) {
    // Example game log header
    this.log(`|game|${this.gameId}`);
    this.log(`|timestamp|${new Date().toISOString()}`);
    playerNames.forEach((name, idx) => {
      this.log(`|player|p${idx + 1}|${name}`);
    });
    this.log(`|gamestart`);

    // Log initial draws
    this.players.forEach((p, idx) => {
      this.log(`|p${idx + 1}|hidden|draw|${p.hand[0]}`);
    });

    // Player 1 starts
    this.log(`|turn|p1`);
  }

  public log(line: string) {
    this.logCallback(line);
    if (this.debug) {
      console.log(`[DEBUG][Engine][${this.gameId}] ${line}`);
    }
  }

  public getCurrentPlayerIndex() {
    return this.currentPlayerIndex;
  }

  public drawCardForCurrentPlayer(): void {
    if (this.gameEnded) return;
    const currentPlayer = this.players[this.currentPlayerIndex];
    if (currentPlayer.eliminated) return;

    if (this.deck.length > 0) {
      const drawn = this.deck.shift();
      if (drawn !== undefined) {
        currentPlayer.hand.push(drawn);
        this.log(`|p${this.currentPlayerIndex + 1}|hidden|draw|${drawn}`);
      }
    } else if (this.removedCard !== null) {
      // If no more cards in deck, draw the removed card
      currentPlayer.hand.push(this.removedCard);
      this.log(`|p${this.currentPlayerIndex + 1}|hidden|draw|${this.removedCard}`);
      this.removedCard = null;
    }
    this.log(`|yourmove|p${this.currentPlayerIndex + 1}`);
  }

  /**
   * makeMove
   * @param playerIndex the index of the player acting
   * @param cardToPlay which card they are discarding
   * @param target optional param; for 2-player, this is usually the other player,
   *               but can also represent a guessed card (for Guard).
   * 
   * Implementation for a standard 2-player game.
   */
  public makeMove(playerIndex: number, cardToPlay: number, target?: number) {
    if (this.gameEnded) return;
    if (playerIndex !== this.currentPlayerIndex) {
      console.log("Invalid move: Not your turn.");
      this.log(`|lose|p${playerIndex + 1}|invalid`);
      this.log(`|end|p${(playerIndex + 1) % 2 + 1}|win`);
      this.gameEnded = true;
      return;
    }

    const currentPlayer = this.players[playerIndex];
    const otherIndex = (playerIndex + 1) % 2;
    const otherPlayer = this.players[otherIndex];

    // Clear Handmaid protection at the start of your turn
    // (since protection lasts until your next turn).
    // But if you want to keep it more literal, do it after the effect finishes.
    // We'll do it at the start of the player's turn:
    currentPlayer.protected = false;

    // Validate card is in hand
    if (!currentPlayer.hand.includes(cardToPlay)) {
      console.log("Invalid move: Card not in hand.");
      this.log(`|lose|p${playerIndex + 1}|invalid`);
      this.log(`|end|p${otherIndex + 1}|win`);
      this.gameEnded = true;
      return;
    }

    // Enforce Countess rule
    if ((currentPlayer.hand.includes(6) || currentPlayer.hand.includes(5)) && 
        currentPlayer.hand.includes(7) && cardToPlay !== 7) {
        console.log("Invalid move: Countess rule violated.");
        this.log(`|lose|p${playerIndex + 1}|invalid`);
        this.log(`|end|p${otherIndex + 1}|win`);
        this.gameEnded = true;
        return;
    }

    // Validate Guard guess
    if (cardToPlay === 1 && ((!otherPlayer.protected && ((target === undefined) || target < 2 || target > 8)) || (otherPlayer.protected && target !== undefined))) {
      console.log("Invalid move: Guard guess out of range or invalid target.");
      this.log(`|lose|p${playerIndex + 1}|invalid`);
      this.log(`|end|p${otherIndex + 1}|win`);
      this.gameEnded = true;
      return;
    }

    // Validate Prince target
    if (cardToPlay === 5 && (target === undefined || target <= 0 || target > this.numPlayers || (target === (otherIndex + 1) && otherPlayer.protected))) {
      console.log("Invalid move: Prince target out of range or invalid.");
      this.log(`|lose|p${playerIndex + 1}|invalid`);
      this.log(`|end|p${otherIndex + 1}|win`);
      this.gameEnded = true;
      return;
    }

    // Play the card
    // For Guard guess, the "target" might represent the guessed card (2..8).
    // For Prince, the "target" might be the index of the other player.
    let playLog = `|p${playerIndex + 1}|play|${cardToPlay}`;
    // If it's Prince, we might specify the target player
    // If it's Guard, we might specify the guessed card
    if (target !== undefined) {
      // Decide if target is a guessed card or a target player
      // We'll guess if it's 2..8 => guard guess
      // If it's 0 or 1 => presumably the other player
      if (cardToPlay === 1) {
        // Guard guess
        playLog += `|${target}`;
      } else if (cardToPlay === 5) {
        // For Prince
        playLog += `|p${target}`;
      }
    }
    this.log(playLog);

    // Remove the card from the player's hand
    const hand = currentPlayer.hand;
    const cardIndex = hand.indexOf(cardToPlay);
    if (cardIndex >= 0) {
      hand.splice(cardIndex, 1);
    }

    // The Countess rule: if you hold Countess (7) with King/Prince you must discard it.
    // Typically enforced by the player, but we won't handle it forcibly here. Just an FYI.

    // Now handle the effect:
    if (cardToPlay === 1 && !otherPlayer.protected) {
      // Guard
      // If the guess is correct and the target has that card => they are eliminated
      // We assume target means the guessed card, and the opponent is the other index
      const guessedCard = target as number; // previously validated
      if (!otherPlayer.eliminated && otherPlayer.hand[0] === guessedCard) {
        // correct guess => other player is eliminated => game ends
        otherPlayer.eliminated = true;
        this.log(`|lose|p${otherIndex + 1}|guard`);
        this.log(`|end|p${playerIndex + 1}|win`);
        this.gameEnded = true;
        return;
      }
    }
    else if (cardToPlay === 2 && !otherPlayer.protected) {
      // Priest => reveal the opponent's card
      if (!otherPlayer.eliminated) {
        const otherCard = otherPlayer.hand[0];
        this.log(`|p${otherIndex + 1}|reveal|${otherCard}`);
      }
    }
    else if (cardToPlay === 3 && !otherPlayer.protected) {
      // Baron => compare hands with the other player
      if (!otherPlayer.eliminated) {
        const myCard = currentPlayer.hand[0];
        const theirCard = otherPlayer.hand[0];
        if (myCard > theirCard) {
          // other player is eliminated
          otherPlayer.eliminated = true;
          this.log(`|lose|p${otherIndex + 1}|baron`);
          this.log(`|end|p${playerIndex + 1}|win`);
          this.gameEnded = true;
          return;
        } else if (theirCard > myCard) {
          // current player is eliminated
          currentPlayer.eliminated = true;
          this.log(`|lose|p${playerIndex + 1}|baron`);
          this.log(`|end|p${otherIndex + 1}|win`);
          this.gameEnded = true;
          return;
        }
        // tie => no effect
      }
    }
    else if (cardToPlay === 4) {
      // Handmaid => protect yourself until your next turn
      currentPlayer.protected = true;
    }
    else if (cardToPlay === 5) {
      // Prince => choose a player (could be self) to discard and draw
      let targetIndex = (target as number) - 1;

      const tPlayer = this.players[targetIndex];
      if (!tPlayer.eliminated) {
        // Discard
        const discardedCard = tPlayer.hand[0];
        this.log(`|p${targetIndex + 1}|discard|${discardedCard}`);
        tPlayer.hand.splice(0, 1);

        // If discarding the Princess => that player is eliminated => game ends
        if (discardedCard === 8) {
          tPlayer.eliminated = true;
          // Prince causes you to lose if you discard Princess
          if (targetIndex === playerIndex) {
            // current player is out
            this.log(`|lose|p${playerIndex + 1}|princess`);
            this.log(`|end|p${otherIndex + 1}|win`);
          } else {
            // the target is out
            this.log(`|lose|p${otherIndex + 1}|princess`);
            this.log(`|end|p${playerIndex + 1}|win`);
          }
          this.gameEnded = true;
          return;
        }

        // Then draw a new card
        if (this.deck.length > 0) {
          const newCard = this.deck.shift();
          if (newCard !== undefined) {
            tPlayer.hand.push(newCard);
            this.log(`|p${targetIndex + 1}|hidden|draw|${newCard}`);
          }
        } else if (this.removedCard !== null) {
          tPlayer.hand.push(this.removedCard);
          this.log(`|p${targetIndex + 1}|hidden|draw|${this.removedCard}`);
          this.removedCard = null;
        }
      }
    }
    else if (cardToPlay === 6 && !otherPlayer.protected) {
      // King => swap hands with the other player
      if (!otherPlayer.eliminated) {
        // Swap
        const myHand = currentPlayer.hand;
        const theirHand = otherPlayer.hand;
        this.log(`|swap|p${playerIndex + 1}|p${otherIndex + 1}|${myHand[0]}|${theirHand[0]}`);

        // In Love Letter each player only holds one card in hand at a time before the draw
        // but if somehow they hold more, we might do a full swap. For simplicity, assume 1.
        currentPlayer.hand = theirHand;
        otherPlayer.hand = myHand;
      }
    }
    else if (cardToPlay === 7) {
      // Countess => no direct effect, aside from the forced discard rule
      // if also holding King(6) or Prince(5). We'll trust the user to handle that.
    }
    else if (cardToPlay === 8) {
      // Princess => if you ever discard it, you lose immediately
      currentPlayer.eliminated = true;
      this.log(`|lose|p${playerIndex + 1}|princess`);
      this.log(`|end|p${otherIndex + 1}|win`);
      this.gameEnded = true;
      return;
    }

    // End of turn, move to next player if game not ended
    // (In 2-player game, just alternate back and forth)
    if (!this.gameEnded) {
      if (this.deck.length === 0) {
        this.log('|nodraw');
        // Compare the two players' hands; highest card wins
        // Ties => the player who went second wins, per the instructions
        const p1Card = this.players[0].hand[0];
        const p2Card = this.players[1].hand[0];
        if (p1Card > p2Card) {
          this.log('|lose|p2|highest');
          this.log(`|end|p1|win`);
        } else if (p2Card > p1Card) {
          this.log('|lose|p1|highest');
          this.log(`|end|p2|win`);
        } else {
          // tie => the player who went second (p2) wins
          this.log('|lose|p1|highest');
          this.log(`|end|p2|win`);
        }
        this.gameEnded = true;
        return;
      }
      this.currentPlayerIndex = (this.currentPlayerIndex + 1) % this.numPlayers;
      // Turn log
      this.log(`|turn|p${otherIndex + 1}`);
    }
  }

  /**
   * isGameOver
   * Checks if the game is over due to:
   *  - a successful elimination (one player left standing)
   *  - no cards left in the deck after a turn
   * If the deck is empty, compare hands. The highest card wins.
   */
  public isGameOver(): boolean {
    return this.gameEnded;
  }
}

