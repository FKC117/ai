module.exports = {
  content: ["./pages/*.{html,js}", "./index.html", "./js/*.js"],
  theme: {
    extend: {
      colors: {
        // Primary Colors - Analytical depth, trustworthy foundation
        primary: {
          DEFAULT: "#1a365d", // blue-900
          50: "#e6f3ff",
          100: "#b3d9ff", 
          200: "#80bfff",
          300: "#4da6ff",
          400: "#1a8cff",
          500: "#0066cc",
          600: "#004d99",
          700: "#003366", // blue-800
          800: "#001a33",
          900: "#1a365d", // blue-900
        },
        // Secondary Colors - Intelligent clarity, breakthrough moments
        secondary: {
          DEFAULT: "#38b2ac", // teal-500
          50: "#e6fffa", // teal-50
          100: "#b2f5ea", // teal-100
          200: "#81e6d9", // teal-200
          300: "#4fd1c7", // teal-300
          400: "#38b2ac", // teal-400
          500: "#319795", // teal-500
          600: "#2c7a7b", // teal-600
          700: "#285e61", // teal-700
          800: "#234e52", // teal-800
          900: "#1d4044", // teal-900
        },
        // Accent Colors - Insight highlighting, warm confidence
        accent: {
          DEFAULT: "#ed8936", // orange-400
          50: "#fffaf0", // orange-50
          100: "#feebc8", // orange-100
          200: "#fbd38d", // orange-200
          300: "#f6ad55", // orange-300
          400: "#ed8936", // orange-400
          500: "#dd6b20", // orange-500
          600: "#c05621", // orange-600
          700: "#9c4221", // orange-700
          800: "#7b341e", // orange-800
          900: "#652b19", // orange-900
        },
        // Background Colors
        background: "#f7fafc", // gray-50 - Clean mental space
        surface: "#edf2f7", // gray-100 - Subtle organization
        
        // Text Colors
        text: {
          primary: "#2d3748", // gray-700 - Extended readability
          secondary: "#718096", // gray-500 - Clear hierarchy
        },
        
        // Status Colors
        success: "#38a169", // green-500 - Positive confirmation
        warning: "#d69e2e", // yellow-500 - Thoughtful caution
        error: "#e53e3e", // red-500 - Helpful correction
        
        // Border Colors
        border: {
          DEFAULT: "#e2e8f0", // gray-200 - Minimal borders
          light: "#f1f5f9", // gray-100 - Subtle separation
        },
      },
      fontFamily: {
        sans: ['Inter', 'sans-serif'],
        mono: ['JetBrains Mono', 'monospace'],
        inter: ['Inter', 'sans-serif'],
        jetbrains: ['JetBrains Mono', 'monospace'],
      },
      fontWeight: {
        normal: '400',
        medium: '500',
        semibold: '600',
        bold: '700',
      },
      boxShadow: {
        'subtle': '0 1px 3px rgba(0, 0, 0, 0.1)', // Gentle elevation for cards
        'primary': '0 4px 12px rgba(26, 54, 93, 0.1)', // Primary shadow for floating elements
        'data-card': '0 2px 8px rgba(26, 54, 93, 0.08)', // Specialized for data visualization cards
      },
      borderRadius: {
        'lg': '0.5rem', // 8px - Standard component radius
        'xl': '0.75rem', // 12px - Card radius
      },
      transitionDuration: {
        '250': '250ms', // State changes
        '400': '400ms', // Data updates
      },
      transitionTimingFunction: {
        'ease-out': 'cubic-bezier(0, 0, 0.2, 1)',
        'ease-in-out': 'cubic-bezier(0.4, 0, 0.2, 1)',
      },
      animation: {
        'data-update': 'dataUpdate 400ms ease-in-out',
        'state-change': 'stateChange 250ms ease-out',
      },
      keyframes: {
        dataUpdate: {
          '0%': { opacity: '0.7', transform: 'translateY(2px)' },
          '100%': { opacity: '1', transform: 'translateY(0)' },
        },
        stateChange: {
          '0%': { opacity: '0.8' },
          '100%': { opacity: '1' },
        },
      },
    },
  },
  plugins: [],
}