// 선 아이콘 — 글자색을 따라간다. 뜻이 있는 아이콘은 옆에 글자를 함께 둔다.
type Props = { size?: number }

const base = (size: number) => ({
  width: size,
  height: size,
  viewBox: '0 0 24 24',
  fill: 'none',
  stroke: 'currentColor',
  strokeWidth: 1.9,
  strokeLinecap: 'round' as const,
  strokeLinejoin: 'round' as const,
  'aria-hidden': true,
})

export const PlusIcon = ({ size = 18 }: Props) => (
  <svg {...base(size)}>
    <path d="M12 5v14M5 12h14" />
  </svg>
)
export const FolderIcon = ({ size = 18 }: Props) => (
  <svg {...base(size)}>
    <path d="M3 7a2 2 0 0 1 2-2h4l2 2h8a2 2 0 0 1 2 2v8a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2z" />
  </svg>
)
export const PauseIcon = ({ size = 18 }: Props) => (
  <svg {...base(size)}>
    <path d="M9 6v12M15 6v12" />
  </svg>
)
export const PlayIcon = ({ size = 18 }: Props) => (
  <svg {...base(size)}>
    <path d="M8 5.5v13l10-6.5z" />
  </svg>
)
export const CheckIcon = ({ size = 18 }: Props) => (
  <svg {...base(size)}>
    <path d="M5 12l5 5L20 7" />
  </svg>
)
export const SettingsIcon = ({ size = 18 }: Props) => (
  <svg {...base(size)}>
    <path d="M4 7h10M18 7h2M4 17h4M12 17h8" />
    <circle cx="16" cy="7" r="2" />
    <circle cx="10" cy="17" r="2" />
  </svg>
)
export const MoreIcon = ({ size = 18 }: Props) => (
  <svg {...base(size)}>
    <circle cx="5" cy="12" r="1.2" />
    <circle cx="12" cy="12" r="1.2" />
    <circle cx="19" cy="12" r="1.2" />
  </svg>
)
export const ChevronLeft = ({ size = 18 }: Props) => (
  <svg {...base(size)}>
    <path d="M15 6l-6 6 6 6" />
  </svg>
)
export const ChevronRight = ({ size = 18 }: Props) => (
  <svg {...base(size)}>
    <path d="M9 6l6 6-6 6" />
  </svg>
)
export const CloseIcon = ({ size = 18 }: Props) => (
  <svg {...base(size)}>
    <path d="M6 6l12 12M18 6L6 18" />
  </svg>
)
export const AlertIcon = ({ size = 18 }: Props) => (
  <svg {...base(size)}>
    <path d="M12 4l9 16H3z" />
    <path d="M12 10v4M12 17h.01" />
  </svg>
)
export const CopyIcon = ({ size = 16 }: Props) => (
  <svg {...base(size)}>
    <rect x="9" y="9" width="11" height="11" rx="2" />
    <path d="M5 15V5a2 2 0 0 1 2-2h8" />
  </svg>
)
