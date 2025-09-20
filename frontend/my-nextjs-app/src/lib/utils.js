import { clsx } from "clsx"
import { twMerge } from "tailwind-merge"

// Merge Tailwind + conditional classNames safely
export function cn(...inputs) {
  return twMerge(clsx(inputs))
}
