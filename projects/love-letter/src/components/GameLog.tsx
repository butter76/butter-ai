import React from 'react';

interface GameLogProps {
	log: string[];
	currentStep: number;
}

const formatLogLine = (line: string): { text: string; highlight: boolean } => {
	const parts = line.split('|');
	if (parts.length < 2) return { text: line, highlight: false };

	switch (parts[1]) {
		case 'play':
			return { 
				text: `${parts[1]} played ${parts[2]}${parts[3] ? ` targeting ${parts[3]}` : ''}`,
				highlight: true 
			};
		case 'draw':
			return { 
				text: `${parts[1]} drew a card`,
				highlight: false 
			};
		case 'reveal':
			return { 
				text: `Card revealed: ${parts[2]}`,
				highlight: true 
			};
		case 'lose':
			return { 
				text: `${parts[1]} lost by ${parts[2]}!`,
				highlight: true 
			};
		case 'end':
			return { 
				text: `Game Over - ${parts[1]} wins!`,
				highlight: true 
			};
		default:
			return { text: line, highlight: false };
	}
};

export const GameLog: React.FC<GameLogProps> = ({ log, currentStep }) => {
	return (
		<div className="bg-gray-50 rounded-lg p-4 max-h-[400px] overflow-y-auto">
			<h3 className="font-bold mb-2 sticky top-0 bg-gray-50 py-2">Game Log</h3>
			<div className="space-y-1">
				{log.slice(0, currentStep + 1).map((line, index) => {
					const { text, highlight } = formatLogLine(line);
					return (
						<div 
							key={index}
							className={`
								font-mono text-sm p-2 rounded transition-all duration-300
								${index === currentStep ? 'bg-blue-100 transform scale-102' : ''}
								${highlight ? 'text-blue-600 font-semibold' : 'text-gray-600'}
							`}
						>
							{text}
						</div>
					);
				})}
			</div>
		</div>
	);
};