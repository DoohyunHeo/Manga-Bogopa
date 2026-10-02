import { useCallback, useEffect, useLayoutEffect, useRef, useState } from 'react'
import { api, pageUrl, post, put, type Book, type Line, type PageLines } from '../api'
import { useApp } from '../App'
import { ConfirmDialog } from '../ConfirmDialog'
import { useDocumentTitle, useFocusTrap } from '../hooks'
import { ChevronLeft, ChevronRight, CloseIcon } from '../icons'

type Props = {
  book: Book
  index: number
  fresh: Map<string, number>
  onIndex: (index: number) => void
  onClose: () => void
}

const isTyping = (target: EventTarget | null) =>
  target instanceof HTMLElement && (target.tagName === 'TEXTAREA' || target.tagName === 'INPUT')

/** 쪽 보기 — 어두운 바탕에 한 쪽. 번역한 쪽은 막대로 원본과 비교하고, 말풍선을 눌러 번역을 그 자리에서 고친다. */
export function Viewer({ book, index, fresh, onIndex, onClose }: Props) {
  const { job, toast } = useApp()
  const pages = book.pages ?? []
  const page = pages[index]
  const [split, setSplit] = useState(0)
  const [ratio, setRatio] = useState(0.707)
  const [lines, setLines] = useState<PageLines | null>(null)
  const [edits, setEdits] = useState<Record<string, string>>({})
  const [openId, setOpenId] = useState<string | null>(null)
  const [draft, setDraft] = useState('')
  const [busy, setBusy] = useState<'' | 'saving' | 'rendering'>('')
  const [needsRender, setNeedsRender] = useState(false)
  const [showSpots, setShowSpots] = useState(false)
  const [leave, setLeave] = useState<null | (() => void)>(null)
  const closeRef = useRef<HTMLButtonElement>(null)
  const rootRef = useRef<HTMLDivElement>(null)
  useFocusTrap(rootRef, !leave) // 확인 창이 떠 있는 동안은 그 창이 초점을 맡는다
  useDocumentTitle(page ? `${index + 1}쪽 — ${book.title}` : book.title)
  const typeset = !!page && (page.typeset || fresh.has(page.name))
  const version = page ? fresh.get(page.name) ?? page.version : 0
  const pending = Object.keys(edits).length
  const bookBusy = job?.status === 'running' && job.book_id === book.id
  const name = page?.name ?? ''

  const loadLines = useCallback(async () => {
    if (!name) return
    try {
      const data = await api<PageLines>(`/api/books/${book.id}/lines/${encodeURIComponent(name)}`)
      setLines(data)
      if (data.size[0] && data.size[1]) setRatio(data.size[0] / data.size[1])
    } catch {
      setLines(null)
    }
  }, [book.id, name])

  useEffect(() => {
    setEdits({})
    setOpenId(null)
    setSplit(0)
    setNeedsRender(false)
    loadLines()
  }, [loadLines])

  // 이 쪽을 다시 쓰는 작업이 끝나면 새 그림·대사를 불러온다
  useEffect(() => {
    if (busy !== 'rendering' || !job || job.status === 'running' || !job.pages.includes(name)) return
    setBusy('')
    loadLines()
    if (job.status === 'done') {
      setNeedsRender(false)
      toast(`${index + 1}쪽을 다시 썼어요.`)
    } else {
      toast(job.message || '이 쪽을 다시 쓰지 못했어요.', 'error')
    }
  }, [job?.status, job?.version]) // eslint-disable-line react-hooks/exhaustive-deps

  const guard = useCallback(
    (action: () => void) => {
      if (pending) setLeave(() => action)
      else action()
    },
    [pending],
  )

  useEffect(() => {
    closeRef.current?.focus()
    document.body.style.overflow = 'hidden'
    return () => {
      document.body.style.overflow = ''
    }
  }, [])

  useEffect(() => {
    const onKey = (e: KeyboardEvent) => {
      if (isTyping(e.target) || leave) return
      if (e.key === 'Escape') {
        if (openId) setOpenId(null)
        else guard(onClose)
      } else if (e.key === 'ArrowRight' && index < pages.length - 1) guard(() => onIndex(index + 1))
      else if (e.key === 'ArrowLeft' && index > 0) guard(() => onIndex(index - 1))
    }
    window.addEventListener('keydown', onKey)
    return () => window.removeEventListener('keydown', onKey)
  }, [index, pages.length, onClose, onIndex, openId, guard, leave])

  if (!page) return null
  const original = pageUrl(book.id, page.name, 'original', 1600)
  const translated = pageUrl(book.id, page.name, 'translated', 1600, version)
  const editable = typeset && !!lines?.has_page && !bookBusy && !busy
  const openLine = lines?.lines.find((line) => line.id === openId) ?? null
  const skipped = lines?.lines.filter((line) => line.status !== 'translated').length ?? 0

  const open = (line: Line) => {
    setOpenId(line.id)
    setDraft(edits[line.id] ?? line.translation)
  }

  const commit = () => {
    if (!openLine) return
    const text = draft.trim()
    const next = { ...edits }
    if (text === openLine.translation.trim()) delete next[openLine.id]
    else next[openLine.id] = text
    setEdits(next)
    setOpenId(null)
  }

  const saveAndRender = async () => {
    if (!lines) return
    setBusy('saving')
    try {
      if (pending) {
        await put(`/api/books/${book.id}/lines/${encodeURIComponent(name)}`, { edits, version: lines.version })
        setEdits({})
        setNeedsRender(true)
      }
      await post(`/api/books/${book.id}/retypeset/${encodeURIComponent(name)}`)
      setBusy('rendering')
    } catch (e) {
      setBusy('')
      toast((e as Error).message, 'error')
      loadLines()
    }
  }

  return (
    <div ref={rootRef} className="viewer" role="dialog" aria-modal="true" aria-label={`${index + 1}쪽 보기`}>
      <div className="stage">
        <div className="topbar">
          <button className="btn small close" ref={closeRef} aria-label="닫기" onClick={() => guard(onClose)}>
            <CloseIcon size={16} />
            <span className="label">닫기</span>
          </button>
          <b>{book.title}</b>
          <span className="where num">
            {index + 1} / {pages.length}쪽{typeset ? '' : ' (아직 번역 전)'}
          </span>
          <span className="spacer" />
          {editable && (
            <>
              <span className="where hint">말풍선을 누르면 번역을 고칠 수 있어요{skipped ? `. 번역하지 않은 글자는 ${skipped}곳이에요` : ''}</span>
              <button className="btn small" aria-pressed={showSpots} onClick={() => setShowSpots((v) => !v)}>
                {showSpots ? '자리 숨기기' : '고칠 자리 보기'}
              </button>
            </>
          )}
          <button className="btn icon" aria-label="이전 쪽" onClick={() => guard(() => onIndex(index - 1))} disabled={index === 0}>
            <ChevronLeft />
          </button>
          <button
            className="btn icon"
            aria-label="다음 쪽"
            onClick={() => guard(() => onIndex(index + 1))}
            disabled={index >= pages.length - 1}
          >
            <ChevronRight />
          </button>
        </div>
        <div className="canvas">
          <div className="frame" style={{ ['--ratio' as string]: String(ratio) }}>
            <img src={original} alt={`${index + 1}쪽 원본`} draggable={false} />
            {typeset && (
              <>
                <img src={translated} alt={`${index + 1}쪽 번역본`} draggable={false} style={{ clipPath: `inset(0 0 0 ${split}%)` }} />
                {split > 0 && <div className="handle" style={{ left: `${split}%` }} />}
                {split > 0 && (
                  <span className="tag" style={{ left: 8 }}>
                    원본
                  </span>
                )}
                {split > 0 && (
                  <span className="tag" style={{ right: 8 }}>
                    번역
                  </span>
                )}
              </>
            )}
            {editable && split === 0 && lines && (
              <div className="spots">
                {lines.lines.map((line) => (
                  <Spot
                    key={line.id}
                    line={line}
                    size={lines.size}
                    edited={line.id in edits}
                    shown={showSpots}
                    active={line.id === openId}
                    onOpen={() => open(line)}
                  />
                ))}
              </div>
            )}
            {openLine && lines && (
              <LineEditor
                line={openLine}
                size={lines.size}
                draft={draft}
                onDraft={setDraft}
                onCancel={() => setOpenId(null)}
                onCommit={commit}
              />
            )}
            {(busy || (bookBusy && job?.mode === 'retypeset' && job.pages.includes(name))) && (
              <div className="rendering" role="status">
                {busy === 'saving' ? '저장하는 중…' : '이 쪽을 다시 쓰는 중…'}
              </div>
            )}
          </div>
          {pending > 0 || needsRender ? (
            <div className="editbar" role="status">
              <span className="num">
                {pending ? `고친 대사 ${pending}개` : '번역은 저장했어요'} — 저장하면 이 쪽만 몇 초 만에 다시 써요
              </span>
              {pending > 0 && (
                <button className="btn small" onClick={() => setEdits({})} aria-disabled={!!busy}>
                  되돌리기
                </button>
              )}
              <button className="btn small primary" onClick={saveAndRender} aria-disabled={!!busy || bookBusy}>
                {busy && <span className="spinner" aria-hidden="true" />}
                {pending ? '저장하고 이 쪽 다시 쓰기' : '이 쪽 다시 쓰기'}
              </button>
            </div>
          ) : typeset ? (
            <label className="compare">
              <span>번역</span>
              <input
                type="range"
                min={0}
                max={100}
                step={1}
                value={split}
                onChange={(e) => {
                  setOpenId(null)
                  setSplit(Number(e.target.value))
                }}
                aria-label="원본과 번역을 비교할 위치 — 오른쪽으로 밀면 원본이 드러나요"
              />
              <span>원본</span>
            </label>
          ) : (
            <p className="empty">이 쪽은 아직 번역하지 않았어요. 번역이 끝나면 여기서 원본과 비교하고 고칠 수 있어요.</p>
          )}
          {bookBusy && typeset && !busy && <p className="empty">이 책을 번역하는 동안에는 고칠 수 없어요. 끝난 뒤에 고쳐 주세요.</p>}
        </div>
      </div>
      {leave && (
        <ConfirmDialog
          title="고친 대사를 저장하지 않고 넘어갈까요?"
          body={`고친 대사 ${pending}개가 사라져요. 남기려면 ‘저장하고 이 쪽 다시 쓰기’를 눌러 주세요.`}
          confirmLabel="버리고 넘어가기"
          danger
          onCancel={() => setLeave(null)}
          onConfirm={() => {
            const action = leave
            setLeave(null)
            setEdits({})
            action()
          }}
        />
      )}
    </div>
  )
}

function boxStyle(box: Line['box'], size: [number, number]) {
  const [x1, y1, x2, y2] = box
  const [w, h] = size
  return { left: `${(x1 / w) * 100}%`, top: `${(y1 / h) * 100}%`, width: `${((x2 - x1) / w) * 100}%`, height: `${((y2 - y1) / h) * 100}%` }
}

function Spot(props: { line: Line; size: [number, number]; edited: boolean; shown: boolean; active: boolean; onOpen: () => void }) {
  const { line, size, edited, shown, active, onOpen } = props
  const where = line.kind === 'bubble' ? '말풍선' : '말풍선 밖 글자'
  const label = line.status === 'translated' ? line.translation : `번역하지 않은 글자: ${line.original}`
  const classes = ['spot', line.status, edited && 'edited', (shown || active) && 'shown'].filter(Boolean).join(' ')
  return (
    <button className={classes} style={boxStyle(line.box, size)} onClick={onOpen} aria-label={`${where} 고치기 — ${label}`}>
      {edited && <span className="badge">고침</span>}
    </button>
  )
}

function LineEditor(props: {
  line: Line
  size: [number, number]
  draft: string
  onDraft: (value: string) => void
  onCancel: () => void
  onCommit: () => void
}) {
  const { line, size, draft, onDraft, onCancel, onCommit } = props
  const ref = useRef<HTMLDivElement>(null)
  const [place, setPlace] = useState<{ left: number; top: number } | null>(null)

  // 말풍선 바로 아래(모자라면 위)에 붙이고, 쪽 밖으로 나가지 않게 당긴다
  useLayoutEffect(() => {
    const measure = () => {
      const box = ref.current
      const frame = box?.parentElement
      if (!box || !frame) return
      const fw = frame.clientWidth
      const fh = frame.clientHeight
      const [x1, y1, , y2] = line.box
      const left = Math.min(Math.max(0, (x1 / size[0]) * fw), Math.max(0, fw - box.offsetWidth))
      const below = (y2 / size[1]) * fh + 8
      const above = (y1 / size[1]) * fh - box.offsetHeight - 8
      const top = below + box.offsetHeight <= fh || above < 0 ? below : above
      setPlace({ left, top: Math.min(top, Math.max(0, fh - box.offsetHeight)) })
    }
    measure()
    window.addEventListener('resize', measure)
    return () => window.removeEventListener('resize', measure)
  }, [line.id, line.box, size])

  const excluded = line.status !== 'translated'
  return (
    <div
      ref={ref}
      className="line-editor"
      role="dialog"
      aria-label="대사 고치기"
      style={place ? { left: place.left, top: place.top } : { visibility: 'hidden' }}
    >
      <div className="le-head">
        <b>{line.kind === 'bubble' ? '말풍선' : '말풍선 밖 글자'}</b>
        {line.note && <span>{line.note}</span>}
      </div>
      <p className="le-original" lang="ja">
        {line.original || '(읽힌 글자가 없어요)'}
      </p>
      <label className="sr-only" htmlFor="line-text">
        번역
      </label>
      <textarea
        id="line-text"
        name="line-text"
        rows={3}
        value={draft}
        autoFocus
        onChange={(e) => onDraft(e.target.value)}
        onKeyDown={(e) => {
          if (e.key === 'Escape') {
            e.preventDefault()
            onCancel()
          } else if (e.key === 'Enter' && (e.ctrlKey || e.metaKey)) {
            e.preventDefault()
            onCommit()
          }
        }}
        placeholder={excluded ? '적어 넣으면 이 글자도 지우고 한국어로 써요…' : '비워 두면 원문을 그대로 둬요…'}
      />
      <div className="le-foot">
        <span>Ctrl+Enter로 확인</span>
        <button className="btn small ghost" onClick={onCancel}>
          그만두기
        </button>
        <button className="btn small primary" onClick={onCommit}>
          확인
        </button>
      </div>
    </div>
  )
}
