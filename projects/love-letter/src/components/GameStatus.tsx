import React from 'react';

interface GameStatusProps {
	currentPlayer: number;
	deckCount: number;
	lastAction?: string;
	gameOver?: boolean;
	winner?: string;
}

export const GameStatus: React.FC<GameStatusProps> = ({
	currentPlayer,
	deckCount,
	lastAction,
	gameOver,
	winner
}) => {
	const getActionDescription = (action: string): string => {
		const parts = action?.split('|') || [];
		if (parts.length < 3) return '';
		const cardNames: Record<string, string> = {
			'1': 'Guard',
			'2': 'Priest',
			'3': 'Baron',
			'4': 'Handmaid',
			'5': 'Prince',
			'6': 'King',
			'7': 'Countess',
			'8': 'Princess'
		};

		switch (parts[2]) {
			case 'play':
				const cardNumber = parts[3];
				return `${parts[1]} played ${cardNames[cardNumber]}`;
			case 'reveal':
				return `Card revealed: ${cardNames[parts[3]]}`;
			case 'discard':
				return `${parts[1]} discarded ${cardNames[parts[3]]}`;
			default:
				return action;
		}
	};

	return (
		<div className="bg-white rounded-lg shadow-md p-4 space-y-2">
			<div className="flex justify-between items-center">
				<div className="flex items-center space-x-2">
					<span className="font-semibold">Current Player:</span>
					<span className={`px-2 py-1 rounded ${
						currentPlayer === 0 ? 'bg-blue-100 text-blue-800' : 'bg-red-100 text-red-800'
					}`}>
						Player {currentPlayer + 1}
					</span>
				</div>
				<div className="flex items-center space-x-2">
					<span className="font-semibold">Deck:</span>
					<span className={`px-2 py-1 rounded ${
						deckCount <= 3 ? 'bg-yellow-100 text-yellow-800' : 'bg-gray-100 text-gray-800'
					}`}>
						{deckCount} cards
					</span>
				</div>
			</div>

			{lastAction && (
				<div className="mt-2 p-2 bg-gray-50 rounded animate-fade-in">
					<p className="text-gray-700">{getActionDescription(lastAction)}</p>
				</div>
			)}

			{gameOver && (
				<div className="mt-4 p-3 bg-green-100 text-green-800 rounded-lg text-center animate-bounce">
					<p className="font-bold">Game Over!</p>
					<p>{winner} wins!</p>
				</div>
			)}
		</div>
	);
};