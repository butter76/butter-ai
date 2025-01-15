import React from 'react';

interface PlaybackControlsProps {
	onPrevious: () => void;
	onNext: () => void;
	onReset: () => void;
	onSpeedChange: (speed: number) => void;
	onPlayPause: () => void;
	isPlaying: boolean;
	currentSpeed: number;
	canGoPrevious: boolean;
	canGoNext: boolean;
	turnCount: number;
	totalTurns: number;
}

export const PlaybackControls: React.FC<PlaybackControlsProps> = ({
	onPrevious,
	onNext,
	onReset,
	onSpeedChange,
	onPlayPause,
	isPlaying,
	currentSpeed,
	canGoPrevious,
	canGoNext,
	turnCount,
	totalTurns,
}) => {
	const speeds = [
		{ label: '0.5x', value: 2000 },
		{ label: '1x', value: 1000 },
		{ label: '2x', value: 500 },
		{ label: '4x', value: 250 },
	];

	return (
		<div className="bg-white rounded-lg shadow-md p-4 space-y-4">
			<div className="flex items-center justify-between">
				<button
					onClick={onReset}
					className="p-2 rounded-full hover:bg-gray-100 transition-colors"
					title="Reset"
				>
					<svg className="w-6 h-6" fill="none" stroke="currentColor" viewBox="0 0 24 24">
						<path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M4 4v5h.582m15.356 2A8.001 8.001 0 004.582 9m0 0H9m11 11v-5h-.581m0 0a8.003 8.003 0 01-15.357-2m15.357 2H15" />
					</svg>
				</button>
				
				<button
					onClick={onPrevious}
					disabled={!canGoPrevious}
					className="p-2 rounded-full hover:bg-gray-100 transition-colors disabled:opacity-50"
					title="Previous"
				>
					<svg className="w-6 h-6" fill="none" stroke="currentColor" viewBox="0 0 24 24">
						<path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M15 19l-7-7 7-7" />
					</svg>
				</button>

				<button
					onClick={onPlayPause}
					className={`p-2 rounded-full transition-colors ${
						isPlaying ? 'bg-red-100 hover:bg-red-200' : 'bg-blue-100 hover:bg-blue-200'
					}`}
					title={isPlaying ? 'Pause' : 'Play'}
				>
					{isPlaying ? (
						<svg className="w-6 h-6" fill="none" stroke="currentColor" viewBox="0 0 24 24">
							<path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M10 9v6m4-6v6m7-3a9 9 0 11-18 0 9 9 0 0118 0z" />
						</svg>
					) : (
						<svg className="w-6 h-6" fill="none" stroke="currentColor" viewBox="0 0 24 24">
							<path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M14.752 11.168l-3.197-2.132A1 1 0 0010 9.87v4.263a1 1 0 001.555.832l3.197-2.132a1 1 0 000-1.664z" />
							<path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M21 12a9 9 0 11-18 0 9 9 0 0118 0z" />
						</svg>
					)}
				</button>

				<button
					onClick={onNext}
					disabled={!canGoNext}
					className="p-2 rounded-full hover:bg-gray-100 transition-colors disabled:opacity-50"
					title="Next"
				>
					<svg className="w-6 h-6" fill="none" stroke="currentColor" viewBox="0 0 24 24">
						<path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9 5l7 7-7 7" />
					</svg>
				</button>

				<select
					value={currentSpeed}
					onChange={(e) => onSpeedChange(Number(e.target.value))}
					className="bg-gray-100 rounded px-2 py-1 text-sm"
				>
					{speeds.map((speed) => (
						<option key={speed.value} value={speed.value}>
							{speed.label}
						</option>
					))}
				</select>
			</div>

			<div className="relative pt-1">
				<div className="flex mb-2 items-center justify-between">
					<div className="text-xs font-semibold inline-block text-blue-600">
						Turn {turnCount} / {totalTurns}
					</div>
				</div>
				<div className="overflow-hidden h-2 mb-4 text-xs flex rounded bg-blue-200">
					<div
						style={{ width: `${(turnCount / totalTurns) * 100}%` }}
						className="shadow-none flex flex-col text-center whitespace-nowrap text-white justify-center bg-blue-500 transition-all duration-300"
					/>
				</div>
			</div>
		</div>
	);
};