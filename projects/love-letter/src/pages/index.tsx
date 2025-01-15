import { useState, useEffect, useCallback } from 'react';
import { Card } from '../components/Card';
import { GameLog } from '../components/GameLog';
import { DiscardPile } from '../components/DiscardPile';
import { GameStatus } from '../components/GameStatus';
import { CardEffect } from '../components/CardEffect';
import { PlaybackControls } from '../components/PlaybackControls';
import { DeckDisplay } from '../components/DeckDisplay';

interface DiscardedCard {
	card: number;
	player: string;
	turn: number;
}

interface GameState {
	p1Name: string;
	p2Name: string;
	currentStep: number;
	log: string[];
	gameId: string;
	timestamp: string;
	currentPlayer: number;
	deckCount: number;
}

const CARD_NAMES = {
	1: 'Guard',
	2: 'Priest',
	3: 'Baron',
	4: 'Handmaid',
	5: 'Prince',
	6: 'King',
	7: 'Countess',
	8: 'Princess'
};

export default function Home() {
	const [gameState, setGameState] = useState<GameState>({
		p1Name: '',
		p2Name: '',
		currentStep: 0,
		log: [],
		gameId: '',
		timestamp: '',
		currentPlayer: 0,
		deckCount: 15, // 16 - 1 removed card
	});

	const [p1Hand, setP1Hand] = useState<number[]>([]);
	const [p2Hand, setP2Hand] = useState<number[]>([]);
	const [p1Protected, setP1Protected] = useState(false);
	const [p2Protected, setP2Protected] = useState(false);
	const [isPlaying, setIsPlaying] = useState(false);
	const [playbackSpeed, setPlaybackSpeed] = useState(1000); // 1 second per move
	const [totalTurns, setTotalTurns] = useState(0);
	const [discardPile, setDiscardPile] = useState<DiscardedCard[]>([]);
	const [turnCount, setTurnCount] = useState(0);
	const [currentEffect, setCurrentEffect] = useState<string | null>(null);
	const [gameOver, setGameOver] = useState(false);
	const [winner, setWinner] = useState<string>('');

	const handleFileUpload = (event: React.ChangeEvent<HTMLInputElement>) => {
		const file = event.target.files?.[0];
		if (file) {
			const reader = new FileReader();
			reader.onload = (e) => {
				const content = e.target?.result as string;
				const lines = content.split('\n').filter(line => line.trim());
				setGameState(prev => ({
					...prev,
					log: lines,
					currentStep: 0
				}));
				// Reset game state
				setP1Hand([]);
				setP2Hand([]);
				setP1Protected(false);
				setP2Protected(false);
				// Parse initial state
				parseGameHeader(lines);
			};
			reader.readAsText(file);
		}
	};

	const parseGameHeader = (log: string[]) => {
		let gameId = '', timestamp = '', p1Name = '', p2Name = '';
		
		log.forEach(line => {
			const parts = line.split('|');
			if (parts[1] === 'game') gameId = parts[2];
			if (parts[1] === 'timestamp') timestamp = parts[2];
			if (parts[1] === 'player') {
				if (parts[2] === 'p1') p1Name = parts[3];
				if (parts[2] === 'p2') p2Name = parts[3];
			}
		});

		setGameState(prev => ({
			...prev,
			gameId,
			timestamp,
			p1Name,
			p2Name
		}));
	};

	const processGameStep = useCallback((step: number) => {
		const line = gameState.log[step];
		const parts = line.split('|');

		// Handle card effects
		if (parts[2] === 'play') {
			const cardNumber = parts[3];
			const effectMap: Record<string, string> = {
				'1': 'guard',
				'2': 'priest',
				'3': 'baron',
				'4': 'handmaid',
				'5': 'prince',
				'6': 'king',
				'7': 'countess',
				'8': 'princess'
			};
			setCurrentEffect(effectMap[cardNumber]);
		}

		// Handle game over conditions
		if (parts[1] === 'end') {
			setGameOver(true);
			setWinner(parts[2]);
		}

		// Handle King's swap action
		if (parts[1] === 'swap') {
			const p1Card = parseInt(parts[4]);
			const p2Card = parseInt(parts[5]);
			setP1Hand([p2Card]);
			setP2Hand([p1Card]);
		}
	
		// Handle Prince's discard action
		if (parts[2] === 'discard') {
			const player = parts[1];
			const discardedCard = parseInt(parts[3]);
			
			if (player === 'p1') {
				setP1Hand(prev => {
					const index = prev.indexOf(discardedCard);
					return index > -1 ? [...prev.slice(0, index), ...prev.slice(index + 1)] : prev;
				});
			} else {
				setP2Hand(prev => {
					const index = prev.indexOf(discardedCard);
					return index > -1 ? [...prev.slice(0, index), ...prev.slice(index + 1)] : prev;
				});
			}			
			// Add to discard pile
			setDiscardPile(prev => [...prev, {
				card: discardedCard,
				player: player,
				turn: turnCount
			}]);
		}

		// Track discards
		if ((parts[1] === 'p1' || parts[1] === 'p2') && parts[2] === 'play') {
			setDiscardPile(prev => [...prev, {
				card: parseInt(parts[3]),
				player: parts[1],
				turn: turnCount
			}]);
			setTurnCount(prev => prev + 1);
		}

		// Update deck count for draw actions
		if ((parts[1] === 'p1' || parts[1] === 'p2') && parts[2] === 'hidden' && parts[3] === 'draw') {
		  setGameState(prev => ({
			...prev,
			deckCount: prev.deckCount - 1
		  }));
		  if (parts[1] === 'p1') {
			setP1Hand(prev => [...prev, parseInt(parts[4])]);
		  } else {
			setP2Hand(prev => [...prev, parseInt(parts[4])]);
		  }
		}

		// Update current player on turn changes
		if (parts[1] === 'turn') {
		  setGameState(prev => ({
			...prev,
			currentPlayer: parts[2] === 'p1' ? 0 : 1
		  }));
		}

		if (parts[1] === 'p1' && parts[2] === 'play') {
		  setP1Hand(prev => {
		    const index = prev.indexOf(parseInt(parts[3]));
		    return index > -1 ? [...prev.slice(0, index), ...prev.slice(index + 1)] : prev;
		  });
		}
		if (parts[1] === 'p2' && parts[2] === 'play') {
		  setP2Hand(prev => {
		    const index = prev.indexOf(parseInt(parts[3]));
		    return index > -1 ? [...prev.slice(0, index), ...prev.slice(index + 1)] : prev;
		  });
		}
		// Handle protection status
		if ((parts[1] === 'p1' || parts[1] === 'p2') && parts[2] === 'play' && parts[3] === '4') {
		  if (parts[1] === 'p1') setP1Protected(true);
		  if (parts[1] === 'p2') setP2Protected(true);
		}
	  }, [gameState.log]);

	const reset = () => {
		setP1Hand([]);
		setP2Hand([]);
		setP1Protected(false);
		setP2Protected(false);
		setDiscardPile([]); // Reset discard pile
		setTurnCount(0); // Reset turn counter
		setCurrentEffect(null); // Reset current effect
		setGameOver(false); // Reset game over state
		setWinner(''); // Reset winner
		setGameState(prev => ({
			...prev,
			currentStep: 0,
			deckCount: 15,
			currentPlayer: 0 // Reset current player
		}));
		setIsPlaying(false);
	};

	useEffect(() => {
		let interval: NodeJS.Timeout | null = null;
		
		if (isPlaying && gameState.currentStep < gameState.log.length - 1) {
			interval = setInterval(() => {
				if (gameState.currentStep < gameState.log.length - 1) {
					nextStep();
				} else {
					setIsPlaying(false);
				}
			}, playbackSpeed);
		}

		return () => {
			if (interval) {
				clearInterval(interval);
			}
		};
	}, [isPlaying, gameState.currentStep, gameState.log.length, playbackSpeed]);

	const nextStep = () => {
		if (gameState.currentStep < gameState.log.length - 1) {
			processGameStep(gameState.currentStep + 1);
			setGameState(prev => ({
				...prev,
				currentStep: prev.currentStep + 1
			}));
		}
	};

	const prevStep = () => {
		if (gameState.currentStep > 0) {
			// Reset all state variables
			setP1Hand([]);
			setP2Hand([]);
			setP1Protected(false);
			setP2Protected(false);
			setDiscardPile([]); // Reset discard pile
			setTurnCount(0); // Reset turn counter
			setCurrentEffect(null); // Reset current effect
			setGameOver(false); // Reset game over state
			setWinner(''); // Reset winner
			setGameState(prev => ({
				...prev,
				deckCount: 15, // Reset deck count
				currentPlayer: 0 // Reset current player
			}));
			
			// Replay all steps up to the target step
			for (let i = 0; i <= gameState.currentStep - 1; i++) {
				processGameStep(i);
			}
			
			// Update the current step
			setGameState(prev => ({
				...prev,
				currentStep: prev.currentStep - 1
			}));
		}
	};

	useEffect(() => {
		// Calculate total turns from log
		const turns = gameState.log.filter(line => line.includes('|turn|')).length;
		setTotalTurns(turns);
	}, [gameState.log]);

	const handleSpeedChange = (speed: number) => {
		setPlaybackSpeed(speed);
	};

	const handlePlayPause = () => {
		setIsPlaying(!isPlaying);
	};

	const handleEffectComplete = () => {
		setCurrentEffect(null);
	};

	return (
		<div className="min-h-screen bg-gray-100 py-6">
			<div className="max-w-7xl mx-auto px-4">
				<div className="bg-white rounded-xl shadow-lg p-6">
					<div className="flex justify-between items-center mb-6">
						<div>
							<h1 className="text-3xl font-bold">Love Letter Game Replay</h1>
							<p className="text-gray-600">Game ID: {gameState.gameId}</p>
						</div>
						<GameStatus
							currentPlayer={gameState.currentPlayer}
							deckCount={gameState.deckCount}
							lastAction={gameState.log[gameState.currentStep]}
							gameOver={gameOver}
							winner={winner}
						/>
						<input
							type="file"
							onChange={handleFileUpload}
							className="block w-full max-w-xs text-sm text-gray-500
								file:mr-4 file:py-2 file:px-4
								file:rounded-full file:border-0
								file:text-sm file:font-semibold
								file:bg-blue-50 file:text-blue-700
								hover:file:bg-blue-100
								transition-all duration-300"
						/>
					</div>

					<div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
						<div className="lg:col-span-2 space-y-6">
							{/* Player 2's hand */}
							<div className="bg-red-50 p-4 rounded-lg transform transition-all duration-300">
								<h2 className="text-xl font-semibold mb-4">{gameState.p2Name} (Player 2)</h2>
								<div className="flex gap-4 flex-wrap justify-center">
									{p2Hand.map((card, i) => (
										<Card
											key={i}
											value={card}
											name={CARD_NAMES[card as keyof typeof CARD_NAMES]}
											isProtected={p2Protected}
											isHighlighted={gameState.currentPlayer === 1}
										/>
									))}
								</div>
							</div>

							{/* Center area with deck and discard */}
							<div className="flex justify-center items-center gap-8">
								<DeckDisplay 
									cardsRemaining={gameState.deckCount}
									isActive={!gameOver}
								/>
								<DiscardPile discards={discardPile} />
							</div>

							{/* Player 1's hand */}
							<div className="bg-blue-50 p-4 rounded-lg transform transition-all duration-300">
								<h2 className="text-xl font-semibold mb-4">{gameState.p1Name} (Player 1)</h2>
								<div className="flex gap-4 flex-wrap justify-center">
									{p1Hand.map((card, i) => (
										<Card
											key={i}
											value={card}
											name={CARD_NAMES[card as keyof typeof CARD_NAMES]}
											isProtected={p1Protected}
											isHighlighted={gameState.currentPlayer === 0}
										/>
									))}
								</div>
							</div>

							{/* Playback Controls */}
							<PlaybackControls
								onPrevious={prevStep}
								onNext={nextStep}
								onReset={reset}
								onSpeedChange={handleSpeedChange}
								onPlayPause={handlePlayPause}
								isPlaying={isPlaying}
								currentSpeed={playbackSpeed}
								canGoPrevious={gameState.currentStep > 0}
								canGoNext={gameState.currentStep < gameState.log.length - 1}
								turnCount={turnCount}
								totalTurns={totalTurns}
							/>
						</div>

						<div>
							<GameLog log={gameState.log} currentStep={gameState.currentStep} />
						</div>
					</div>
				</div>
			</div>
			{currentEffect && (
				<div className="pointer-events-none fixed inset-0 flex items-center justify-center z-10">
					<CardEffect
						effect={currentEffect}
						onComplete={handleEffectComplete}
					/>
				</div>
			)}
		</div>
	);
}