import React from 'react';
import { Card } from './Card';

interface GameBoardProps {
	p1Name: string;
	p2Name: string;
	p1Hand: number[];
	p2Hand: number[];
	p1Protected: boolean;
	p2Protected: boolean;
	currentPlayer: number;
	deckCount: number;
}

export const GameBoard: React.FC<GameBoardProps> = ({
	p1Name,
	p2Name,
	p1Hand,
	p2Hand,
	p1Protected,
	p2Protected,
	currentPlayer,
	deckCount,
}) => {
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

	return (
		<div className="relative w-full max-w-4xl mx-auto h-[600px] bg-green-800 rounded-xl p-8">
			{/* Player 2 Area */}
			<div className={`absolute top-4 left-1/2 transform -translate-x-1/2 transition-all ${
				currentPlayer === 1 ? 'scale-105' : ''
			}`}>
				<div className="text-white text-center mb-2">
					<span className="font-bold">{p2Name}</span>
					{p2Protected && (
						<span className="ml-2 px-2 py-1 bg-blue-500 rounded-full text-xs">Protected</span>
					)}
				</div>
				<div className="flex justify-center gap-4">
					{p2Hand.map((card, i) => (
						<Card
							key={i}
							value={card}
							name={CARD_NAMES[card as keyof typeof CARD_NAMES]}
							isProtected={p2Protected}
						/>
					))}
				</div>
			</div>

			{/* Deck Area */}
			<div className="absolute top-1/2 left-1/2 transform -translate-x-1/2 -translate-y-1/2">
				<div className="relative">
					{[...Array(Math.min(3, deckCount))].map((_, i) => (
						<div
							key={i}
							className="absolute bg-red-800 w-24 h-36 rounded-lg border-2 border-gold shadow-xl transform"
							style={{
								transform: `translateX(${i * 2}px) translateY(${i * 2}px) rotate(${i * 2}deg)`,
							}}
						/>
					))}
					<div className="text-white text-center mt-40">
						{deckCount} cards remaining
					</div>
				</div>
			</div>

			{/* Player 1 Area */}
			<div className={`absolute bottom-4 left-1/2 transform -translate-x-1/2 transition-all ${
				currentPlayer === 0 ? 'scale-105' : ''
			}`}>
				<div className="flex justify-center gap-4 mb-2">
					{p1Hand.map((card, i) => (
						<Card
							key={i}
							value={card}
							name={CARD_NAMES[card as keyof typeof CARD_NAMES]}
							isProtected={p1Protected}
						/>
					))}
				</div>
				<div className="text-white text-center">
					<span className="font-bold">{p1Name}</span>
					{p1Protected && (
						<span className="ml-2 px-2 py-1 bg-blue-500 rounded-full text-xs">Protected</span>
					)}
				</div>
			</div>
		</div>
	);
};