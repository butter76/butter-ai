import React from 'react';
import { Card } from './Card';

interface DiscardPileProps {
	discards: Array<{
		card: number;
		player: string;
		turn: number;
	}>;
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

export const DiscardPile: React.FC<DiscardPileProps> = ({ discards }) => {
	return (
		<div className="bg-gray-50 rounded-lg p-4">
			<h3 className="font-bold mb-2">Discard Pile</h3>
			<div className="flex flex-wrap gap-2">
				{discards.map((discard, index) => (
					<div 
						key={index}
						className="relative"
						style={{
							transform: `rotate(${Math.random() * 20 - 10}deg)`,
							transition: 'transform 0.3s ease'
						}}
					>
						<div className="absolute -top-2 -right-2 bg-gray-800 text-white text-xs px-2 py-1 rounded-full">
							{discard.player}
						</div>
						<Card
							value={discard.card}
							name={CARD_NAMES[discard.card as keyof typeof CARD_NAMES]}
							isSmall
						/>
					</div>
				))}
			</div>
		</div>
	);
};