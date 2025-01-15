import React, { useEffect, useState } from 'react';

interface CardEffectProps {
	effect: string;
	onComplete: () => void;
}

export const CardEffect: React.FC<CardEffectProps> = ({ effect, onComplete }) => {
	const [visible, setVisible] = useState(true);

	useEffect(() => {
		const timer = setTimeout(() => {
			setVisible(false);
			onComplete();
		}, 2000);

		return () => clearTimeout(timer);
	}, [effect, onComplete]);

	if (!visible) return null;

	const getEffectStyle = () => {
		switch (effect) {
			case 'guard':
				return 'bg-red-500 text-white';
			case 'priest':
				return 'bg-blue-500 text-white';
			case 'baron':
				return 'bg-green-500 text-white';
			case 'handmaid':
				return 'bg-yellow-500 text-black';
			case 'prince':
				return 'bg-purple-500 text-white';
			case 'king':
				return 'bg-orange-500 text-white';
			case 'countess':
				return 'bg-pink-500 text-white';
			case 'princess':
				return 'bg-indigo-500 text-white';
			default:
				return 'bg-gray-500 text-white';
		}
	};

	return (
		<div className="fixed inset-0 flex items-center justify-center z-50">
			<div className={`
				${getEffectStyle()}
				p-6 rounded-lg shadow-lg
				animate-bounce
				transform transition-all duration-500
			`}>
				<p className="text-2xl font-bold">{effect.toUpperCase()} EFFECT!</p>
			</div>
		</div>
	);
};