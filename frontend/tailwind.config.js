/** @type {import('tailwindcss').Config} */
export default {
  content: ["./index.html", "./src/**/*.{js,ts,jsx,tsx}"],
  theme: {
    extend: {
      colors: {
        ink: "#10233f",
        teal: { 50: "#effcfb", 100: "#d9f7f4", 400: "#52c8c2", 500: "#22aaa5", 600: "#158d8b", 700: "#0d6f71" },
        mist: "#f6f9fc",
        line: "#e5edf3",
      },
      fontFamily: { sans: ["Inter", "ui-sans-serif", "system-ui", "sans-serif"] },
      boxShadow: {
        soft: "0 14px 40px rgba(16,35,63,0.06)",
        lift: "0 18px 45px rgba(16,35,63,0.12)",
      },
    },
  },
  plugins: [],
};
