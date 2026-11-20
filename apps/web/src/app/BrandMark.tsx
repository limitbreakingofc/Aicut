import { useId } from "react";

/** Aicut logo: a "play" glyph sliced by a cut, drawn in the accent-foreground color. */
export function BrandMark({ size = 20 }: { size?: number }) {
  const maskId = useId();
  return (
    <span className="brand-mark" style={{ width: size, height: size }} aria-hidden>
      <svg width={size * 0.66} height={size * 0.66} viewBox="0 0 24 24">
        <defs>
          <mask id={maskId}>
            <rect width="24" height="24" fill="#fff" />
            <path d="M5 17.6 L19.5 7" stroke="#000" strokeWidth="2.6" strokeLinecap="round" />
          </mask>
        </defs>
        <path
          d="M8 5.4 L19 12 L8 18.6 Z"
          fill="currentColor"
          stroke="currentColor"
          strokeWidth="2.4"
          strokeLinejoin="round"
          mask={`url(#${maskId})`}
        />
      </svg>
    </span>
  );
}
