import React from 'react';

interface CardProps {
	value: number;
	name: string;
	isProtected?: boolean;
	isSmall?: boolean;
	isHighlighted?: boolean;
}

const CARD_COLORS = {
	1: 'bg-gray-200',
	2: 'bg-blue-200',
	3: 'bg-green-200',
	4: 'bg-yellow-200',
	5: 'bg-purple-200',
	6: 'bg-red-200',
	7: 'bg-pink-200',
	8: 'bg-indigo-200',
};

const CARD_DESCRIPTIONS = {
	1: 'Guess opponent\'s card (2-8)',
	2: 'Look at opponent\'s hand',
	3: 'Compare hands with opponent',
	4: 'Protection until next turn',
	5: 'Force player to discard and draw',
	6: 'Trade hands with opponent',
	7: 'Must discard if with King/Prince',
	8: 'Lose if discarded',
};

export const Card: React.FC<CardProps> = ({ 
	value, 
	name, 
	isProtected, 
	isSmall,
	isHighlighted 
}) => {
	return (
		<div 
			className={`
				relative rounded-lg shadow-md 
				${CARD_COLORS[value as keyof typeof CARD_COLORS]} 
				${isSmall ? 'p-2 min-w-[80px]' : 'p-4 min-w-[120px]'}
				${isHighlighted ? 'ring-2 ring-yellow-400 transform scale-105' : ''}
				transition-all duration-300 ease-in-out
				hover:shadow-lg hover:-translate-y-1
			`}
		>
			<div className={`absolute top-2 left-2 font-bold ${isSmall ? 'text-base' : 'text-lg'}`}>
				{value}
			</div>
			<div className={`text-center ${isSmall ? 'mt-2 text-sm' : 'mt-4'} font-semibold`}>
				{name}
			</div>
			{!isSmall && (
				<div className="text-xs mt-2 text-gray-600 text-center">
					{CARD_DESCRIPTIONS[value as keyof typeof CARD_DESCRIPTIONS]}
				</div>
			)}
			{isProtected && (
				<div className="absolute top-0 right-0 w-full h-full flex items-center justify-center bg-blue-500 bg-opacity-30 rounded-lg backdrop-blur-sm">
					<span className="text-blue-800 font-bold animate-pulse">Protected</span>
				</div>
			)}
		</div>
	);
};