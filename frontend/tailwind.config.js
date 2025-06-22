/** @type {import('tailwindcss').Config} */
export default {
  content: [
    "./index.html",
    "./src/**/*.{vue,js,ts,jsx,tsx}",
  ],
  darkMode: 'class',
  theme: {
    extend: {
      colors: {
        // Dark Mode Colors (Default/Primary)
        dark: {
          bg: '#0f1117',        // Deep charcoal base
          surface: '#1a1d24',   // UI blocks, cards, modals
          border: '#2a2d36',    // Dividers, cards, subtle edges
          text: {
            primary: '#eaeaea',  // Readable on dark
            secondary: '#999999', // Subdued content or labels
          }
        },
        // Light Mode Colors
        light: {
          bg: '#ffffff',        // White background
          surface: '#f3f4f6',   // Panels
          border: '#d1d5db',    // Borders
          text: {
            primary: '#1a1a1a',  // Main text
            secondary: '#6b7280', // Secondary text
          }
        },
        // Accent Colors (same for both modes but different usage)
        accent: {
          primary: {
            dark: '#00b3ff',    // Electric blue for dark mode
            light: '#007aff',   // Blue for light mode
          },
          secondary: {
            dark: '#26e6a6',    // Emerald for dark mode
            light: '#00c896',   // Teal for light mode
          },
          error: {
            dark: '#ff6b6b',    // Red for dark mode
            light: '#ef4444',   // Red for light mode
          }
        },
        // Convenience mappings for existing Tailwind classes
        primary: {
          50: '#e6f7ff',
          100: '#bae7ff',
          200: '#7dd3fc',
          300: '#38bdf8',
          400: '#0ea5e9',
          500: '#00b3ff',  // Dark mode primary
          600: '#0284c7',
          700: '#0369a1',
          800: '#075985',
          900: '#0c4a6e',
        },
        secondary: {
          50: '#ecfdf5',
          100: '#d1fae5',
          200: '#a7f3d0',
          300: '#6ee7b7',
          400: '#26e6a6',  // Dark mode secondary
          500: '#10b981',
          600: '#059669',
          700: '#047857',
          800: '#065f46',
          900: '#064e3b',
        },
        error: {
          50: '#fef2f2',
          100: '#fee2e2',
          200: '#fecaca',
          300: '#fca5a5',
          400: '#ff6b6b',  // Dark mode error
          500: '#ef4444',
          600: '#dc2626',
          700: '#b91c1c',
          800: '#991b1b',
          900: '#7f1d1d',
        }
      },
      fontFamily: {
        'sans': ['Inter', 'system-ui', 'sans-serif'],
        'mono': ['JetBrains Mono', 'Monaco', 'Consolas', 'monospace']
      },
      animation: {
        'fade-in': 'fadeIn 0.2s ease-in-out',
        'slide-in': 'slideIn 0.3s ease-out',
        'pulse-subtle': 'pulse 2s cubic-bezier(0.4, 0, 0.6, 1) infinite',
      },
      keyframes: {
        fadeIn: {
          '0%': { opacity: '0' },
          '100%': { opacity: '1' },
        },
        slideIn: {
          '0%': { transform: 'translateY(-10px)', opacity: '0' },
          '100%': { transform: 'translateY(0)', opacity: '1' },
        },
      },
    },
  },
  plugins: [],
}
