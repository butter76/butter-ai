import React from 'react';

interface ControlsProps {
	onPrevious: () => void;
	onNext: () => void;
	onReset: () => void;
	onPlayPause: () => void;
	isPlaying: boolean;
	canGoPrevious: boolean;
	canGoNext: boolean;
}

export const Controls: React.FC<ControlsProps> = ({
	onPrevious,
	onNext,
	onReset,
	onPlayPause,
	isPlaying,
	canGoPrevious,
	canGoNext,
}) => {
	return (
		<div className="flex items-center justify-center gap-4 p-4 bg-white rounded-lg shadow-md">
			<button
				onClick={onReset}
				className="px-4 py-2 bg-gray-200 rounded hover:bg-gray-300 transition-colors"
			>
				<svg className="w-5 h-5" fill="none" stroke="currentColor" viewBox="0 0 24 24">
					<path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M4 4v5h.582m15.356 2A8.001 8.001 0 004.582 9m0 0H9m11 11v-5h-.581m0 0a8.003 8.003 0 01-15.357-2m15.357 2H15" />
				</svg>
			</button>
			
			<button
				onClick={onPrevious}
				disabled={!canGoPrevious}
				className="px-4 py-2 bg-gray-200 rounded hover:bg-gray-300 transition-colors disabled:opacity-50"
			>
				<svg className="w-5 h-5" fill="none" stroke="currentColor" viewBox="0 0 24 24">
					<path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M15 19l-7-7 7-7" />
				</svg>
			</button>
			
			<button
				onClick={onPlayPause}
				className="px-4 py-2 bg-blue-500 text-white rounded hover:bg-blue-600 transition-colors"
			>
				{isPlaying ? (
					<svg className="w-5 h-5" fill="none" stroke="currentColor" viewBox="0 0 24 24">
						<path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M10 9v6m4-6v6m7-3a9 9 0 11-18 0 9 9 0 0118 0z" />
					</svg>
				) : (
					<svg className="w-5 h-5" fill="none" stroke="currentColor" viewBox="0 0 24 24">
						<path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M14.752 11.168l-3.197-2.132A1 1 0 0010 9.87v4.263a1 1 0 001.555.832l3.197-2.132a1 1 0 000-1.664z" />
						<path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M21 12a9 9 0 11-18 0 9 9 0 0118 0z" />
					</svg>
				)}
			</button>
			
			<button
				onClick={onNext}
				disabled={!canGoNext}
				className="px-4 py-2 bg-gray-200 rounded hover:bg-gray-300 transition-colors disabled:opacity-50"
			>
				<svg className="w-5 h-5" fill="none" stroke="currentColor" viewBox="0 0 24 24">
					<path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9 5l7 7-7 7" />
				</svg>
			</button>
		</div>
	);
};