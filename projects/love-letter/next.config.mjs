/** @type {import('next').NextConfig} */
const nextConfig = {
	reactStrictMode: true,
	pageExtensions: ['tsx', 'ts'],
	webpack: (config) => {
		config.resolve.fallback = {
			fs: false,
			path: false,
		};
		return config;
	},
};

export default nextConfig;