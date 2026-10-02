import { useEffect, useRef } from 'react'
import { useFocusTrap } from './hooks'

type Props = {
  title: string
  body: string
  confirmLabel: string
  danger?: boolean
  onCancel: () => void
  onConfirm: () => void
}

/** 되돌리기 어려운 일을 하기 전에 한 번 묻는 창. Esc·바깥 누르기 = 그만두기. */
export function ConfirmDialog({ title, body, confirmLabel, danger, onCancel, onConfirm }: Props) {
  const cancelRef = useRef<HTMLButtonElement>(null)
  const dialogRef = useRef<HTMLDivElement>(null)
  useFocusTrap(dialogRef)
  useEffect(() => {
    cancelRef.current?.focus()
    const onKey = (e: KeyboardEvent) => {
      if (e.key === 'Escape') {
        e.stopPropagation()
        onCancel()
      }
    }
    window.addEventListener('keydown', onKey, true)
    return () => window.removeEventListener('keydown', onKey, true)
  }, [onCancel])
  return (
    <div className="scrim" onClick={onCancel}>
      <div ref={dialogRef} className="dialog" role="dialog" aria-modal="true" aria-labelledby="confirm-title" onClick={(e) => e.stopPropagation()}>
        <h2 id="confirm-title">{title}</h2>
        <p>{body}</p>
        <div className="row">
          <button className="btn" ref={cancelRef} onClick={onCancel}>
            그만두기
          </button>
          <button className={`btn primary${danger ? ' danger' : ''}`} onClick={onConfirm}>
            {confirmLabel}
          </button>
        </div>
      </div>
    </div>
  )
}
