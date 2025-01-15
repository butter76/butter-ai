import React from 'react';

interface DeckDisplayProps {
	cardsRemaining: number;
	isActive: boolean;
}

export const DeckDisplay: React.FC<DeckDisplayProps> = ({ cardsRemaining, isActive }) => {
	return (
		<div className="relative w-32 h-48">
			{/* Stack effect for remaining cards */}
			{[...Array(Math.min(5, cardsRemaining))].map((_, i) => (
				<div
					key={i}
					className={`absolute rounded-lg border-2 border-gray-300 shadow-md
						${isActive ? 'bg-red-600' : 'bg-red-800'}
						transition-all duration-300`}
					style={{
						width: '100%',
						height: '100%',
						transform: `rotate(${(i - 2) * 2}deg) translateY(${i * 0.5}px)`,
						zIndex: i,
					}}
				>
					<div className="absolute inset-0 flex items-center justify-center">
						<div className="w-16 h-16 rounded-full border-4 border-gray-300 flex items-center justify-center">
							<span className="text-white font-bold text-xl">LL</span>
						</div>
					</div>
				</div>
			))}
			
			<div className="absolute -bottom-8 left-0 right-0 text-center">
				<span className={`font-semibold ${cardsRemaining <= 3 ? 'text-red-600' : 'text-gray-600'}`}>
					{cardsRemaining} cards left
				</span>
			</div>
		</div>
	);
};