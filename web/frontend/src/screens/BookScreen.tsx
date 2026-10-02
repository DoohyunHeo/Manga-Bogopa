import { useCallback, useEffect, useMemo, useRef, useState } from 'react'
import { api, pageUrl, post, type Book, type Job, type RunMode } from '../api'
import { useApp } from '../App'
import { ConfirmDialog } from '../ConfirmDialog'
import { formatDuration, go, useDocumentTitle, useNow } from '../hooks'
import { AlertIcon, CheckIcon, FolderIcon, MoreIcon, PauseIcon, PlayIcon } from '../icons'
import { stoppingLine, translationLeft } from '../StatusBar'
import { Viewer } from './Viewer'

type Confirm = null | 'fresh' | 'remove'

export function BookScreen({ id }: { id: string }) {
  const { job, toast, refreshBooks } = useApp()
  const [book, setBook] = useState<Book | null>(null)
  const [error, setError] = useState('')
  const [menuOpen, setMenuOpen] = useState(false)
  const [confirm, setConfirm] = useState<Confirm>(null)
  const [viewing, setViewing] = useState<number | null>(null)
  const [starting, setStarting] = useState(false)

  const load = useCallback(async () => {
    try {
      setBook(await api<Book>(`/api/books/${id}`))
      setError('')
    } catch (e) {
      setError((e as Error).message)
    }
  }, [id])

  useEffect(() => {
    load()
    // 탐색기에서 파일을 바꾸고 돌아왔을 때도 지금 상태를 보여 준다
    const onFocus = () => document.visibilityState === 'visible' && load()
    window.addEventListener('focus', onFocus)
    document.addEventListener('visibilitychange', onFocus)
    return () => {
      window.removeEventListener('focus', onFocus)
      document.removeEventListener('visibilitychange', onFocus)
    }
  }, [load])

  const mine = job && job.book_id === id ? job : null
  const running = mine?.status === 'running'
  const otherRunning = job && job.status === 'running' && job.book_id !== id

  // 이 책의 작업이 시작·끝나면 책 상태를 다시 읽는다
  useEffect(() => {
    if (mine) load()
  }, [mine?.status, load]) // eslint-disable-line react-hooks/exhaustive-deps

  // 이번 작업에서 새로 끝난 쪽 → 그림을 새로 불러올 버전. 한 쪽을 두 번 쓰면(의심 대사 확인 답으로 다시 쓴 쪽) 뒤의 것이
  // 남는다 — 같은 1초 안에 두 번 써도 버전이 달라지게 1/1000초까지 둔다
  const fresh = useMemo(() => new Map((mine?.done_pages ?? []).map((p) => [p.name, Math.round(p.at * 1000) / 1000])), [mine?.done_pages])

  useDocumentTitle(book?.title ?? '')

  // 메뉴: 열면 첫 항목으로 초점, 바깥을 누르거나 Esc면 닫고 메뉴 버튼으로 초점을 돌린다
  const menuRef = useRef<HTMLDivElement>(null)
  const menuButtonRef = useRef<HTMLButtonElement>(null)
  useEffect(() => {
    if (!menuOpen) return
    menuRef.current?.querySelector<HTMLButtonElement>('button:not([disabled])')?.focus()
    const close = (focusButton: boolean) => {
      setMenuOpen(false)
      if (focusButton) menuButtonRef.current?.focus()
    }
    const onPointer = (e: PointerEvent) => {
      const target = e.target as Node
      if (!menuRef.current?.contains(target) && !menuButtonRef.current?.contains(target)) close(false)
    }
    const onKey = (e: KeyboardEvent) => {
      if (e.key === 'Escape') close(true)
      if (e.key === 'ArrowDown' || e.key === 'ArrowUp') {
        const items = Array.from(menuRef.current?.querySelectorAll<HTMLButtonElement>('button:not([disabled])') ?? [])
        const at = items.indexOf(document.activeElement as HTMLButtonElement)
        items[(at + (e.key === 'ArrowDown' ? 1 : items.length - 1)) % items.length]?.focus()
        e.preventDefault()
      }
    }
    document.addEventListener('pointerdown', onPointer)
    document.addEventListener('keydown', onKey)
    return () => {
      document.removeEventListener('pointerdown', onPointer)
      document.removeEventListener('keydown', onKey)
    }
  }, [menuOpen])

  if (error) {
    return (
      <div className="notice error" role="alert">
        <AlertIcon />
        <span className="grow">{error}</span>
        <button className="btn small" onClick={() => go('')}>
          책장으로
        </button>
      </div>
    )
  }
  if (!book) return <p style={{ color: 'var(--tone)' }}>불러오는 중…</p>

  const pages = book.pages ?? []
  const isTypeset = (name: string, typeset: boolean) => typeset || fresh.has(name)
  const currentIndex =
    running && mine?.stage === 'typesetting' ? pages.findIndex((p) => !isTypeset(p.name, p.typeset)) : -1

  const start = async (mode: RunMode) => {
    setMenuOpen(false)
    setStarting(true)
    try {
      await post(`/api/books/${id}/run`, { mode })
      refreshBooks()
    } catch (e) {
      toast((e as Error).message, 'error')
    } finally {
      setStarting(false)
    }
  }

  const stop = async () => {
    try {
      await post('/api/job/stop') // '멈추는 중'은 진행 상황 칸에 바로 보인다
    } catch (e) {
      toast((e as Error).message, 'error')
    }
  }

  const openFolder = async (which: 'output' | 'input') => {
    setMenuOpen(false)
    try {
      await post(`/api/books/${id}/open`, { which })
    } catch (e) {
      toast((e as Error).message, 'error')
    }
  }

  const remove = async () => {
    setConfirm(null)
    try {
      await api(`/api/books/${id}`, { method: 'DELETE' })
      await refreshBooks()
      go('')
    } catch (e) {
      toast((e as Error).message, 'error')
    }
  }

  let primary = null
  if (running) {
    primary = (
      <button className="btn" onClick={stop} aria-disabled={mine?.stopping}>
        {mine?.stopping ? <span className="spinner" aria-hidden="true" /> : <PauseIcon />}
        {mine?.stopping ? '멈추는 중…' : '멈추기'}
      </button>
    )
  } else if (book.status === 'done') {
    primary = (
      <button className="btn primary" onClick={() => openFolder('output')}>
        <FolderIcon />
        결과 폴더 열기
      </button>
    )
  } else if (book.status !== 'missing') {
    const label = book.status === 'new' ? '번역 시작' : '이어서 번역'
    primary = (
      <button className="btn primary" onClick={() => start('resume')} aria-disabled={starting || !!otherRunning}>
        {starting ? <span className="spinner" aria-hidden="true" /> : <PlayIcon />}
        {label}
      </button>
    )
  }

  return (
    <>
      <header className="book-head">
        <div className="titles">
          <h1>{book.title}</h1>
          <div className="facts">
            <span className="num">{book.page_count}쪽</span>
            <span className="path">결과: {book.output_dir}</span>
          </div>
        </div>
        <div className="actions">
          {book.status !== 'done' && book.typeset_count > 0 && !running && (
            <button className="btn" onClick={() => openFolder('output')}>
              <FolderIcon />
              결과 폴더 열기
            </button>
          )}
          {primary}
          <button
            ref={menuButtonRef}
            className="btn icon"
            aria-label="더 보기"
            aria-haspopup="menu"
            aria-expanded={menuOpen}
            onClick={() => setMenuOpen((open) => !open)}
          >
            <MoreIcon />
          </button>
          {menuOpen && (
            <div className="menu" role="menu" ref={menuRef}>
              {!running && book.translated_count > 0 && (
                <button role="menuitem" onClick={() => start('retypeset')} disabled={!!otherRunning}>
                  글자만 다시 쓰기 (번역은 그대로)
                </button>
              )}
              {!running && book.status !== 'new' && (
                <button role="menuitem" onClick={() => { setMenuOpen(false); setConfirm('fresh') }} disabled={!!otherRunning}>
                  처음부터 다시 번역…
                </button>
              )}
              <button role="menuitem" onClick={() => openFolder('input')}>
                원고 폴더 열기
              </button>
              {!running && (
                <button role="menuitem" className="danger" onClick={() => { setMenuOpen(false); setConfirm('remove') }}>
                  책장에서 빼기…
                </button>
              )}
            </div>
          )}
        </div>
      </header>

      {book.status === 'missing' && (
        <div className="notice error" role="alert">
          <AlertIcon />
          <span className="grow">원고 폴더를 찾지 못했어요: {book.input_dir} — 폴더를 옮겼다면 새 책으로 다시 열어 주세요.</span>
        </div>
      )}
      {otherRunning && (
        <div className="notice info">
          <AlertIcon />
          <span className="grow">
            지금 ‘{job?.title}’을(를) 번역하는 중이에요. 한 번에 한 권씩 번역해요.{' '}
            <a className="link" href={`#/book/${job?.book_id}`}>
              그 책 보기
            </a>
          </span>
        </div>
      )}

      {mine && <ProgressPanel job={mine} book={book} />}

      <section aria-label="쪽 목록">
        <div className="page-grid">
          {pages.map((page, index) => {
            const typeset = isTypeset(page.name, page.typeset)
            const version = fresh.get(page.name) ?? page.version
            const current = index === currentIndex
            return (
              <button
                key={page.name}
                className={`page-tile${current ? ' current' : ''}`}
                onClick={() => setViewing(index)}
                aria-label={`${index + 1}쪽 ${typeset ? '번역본' : '원본'} 크게 보기`}
              >
                <div className="sheet">
                  <img
                    key={typeset ? `t${version}` : 'o'}
                    className={running && Date.now() / 1000 - (fresh.get(page.name) ?? 0) < 10 ? 'fresh' : undefined}
                    src={pageUrl(book.id, page.name, typeset ? 'translated' : 'original', 320, typeset ? version : 0)}
                    alt=""
                    loading="lazy"
                    decoding="async"
                  />
                  {!typeset && <div className="veil" />}
                </div>
                <span className="caption num">{current ? `${index + 1}쪽 쓰는 중` : index + 1}</span>
              </button>
            )
          })}
        </div>
      </section>

      {viewing !== null && (
        <Viewer
          book={book}
          index={viewing}
          fresh={fresh}
          onIndex={setViewing}
          onClose={() => setViewing(null)}
        />
      )}

      {confirm && (
        <ConfirmDialog
          title={confirm === 'fresh' ? '처음부터 다시 번역할까요?' : '책장에서 뺄까요?'}
          body={
            confirm === 'fresh'
              ? '지금 번역은 결과 폴더에 날짜를 붙여 백업해 두고, 글자 찾기부터 번역까지 다시 해요. 번역 요청을 다시 보내므로 시간이 걸려요.'
              : '책장 목록에서만 빼요. 원고 폴더와 결과 폴더의 파일은 그대로 남아요.'
          }
          confirmLabel={confirm === 'fresh' ? '처음부터 다시 번역' : '책장에서 빼기'}
          danger={confirm === 'remove'}
          onCancel={() => setConfirm(null)}
          onConfirm={() => (confirm === 'fresh' ? (setConfirm(null), start('fresh')) : remove())}
        />
      )}
    </>
  )
}

function ProgressPanel({ job, book }: { job: Job; book: Book }) {
  const running = job.status === 'running'
  const now = useNow(running)
  const retypeset = job.mode === 'retypeset'
  const order = ['reading', 'translating', 'typesetting']
  const stageIndex = running ? Math.max(0, order.indexOf(job.stage === 'preparing' ? 'reading' : job.stage)) : 3

  if (!running) return <ResultNotice job={job} book={book} />
  // 서버 알림 사이에도 시간이 흐르게 — 마지막 알림을 받은 뒤 지난 시간을 더한다
  const drift = Math.max(0, (now - (job.received_at ?? now)) / 1000)
  const elapsed = job.elapsed_sec + drift
  const stageElapsed = job.stage_elapsed_sec + drift
  const eta = job.eta_sec !== null ? Math.max(0, job.eta_sec - drift) : null

  // 쪽을 읽는 동안에도 읽은 묶음부터 뒤에서 번역한다 — 그때는 '읽기'와 '번역'이 함께 도는 중
  const dl = job.stage === 'preparing' ? job.download : null // 처음 한 번 모델 파일을 받는 중
  const tr = job.translation
  const translatingAlongside = job.stage === 'reading' && !!tr && tr.total > 0
  const left = translationLeft(job)
  const translateSub =
    job.stage === 'translating'
      ? left
        ? `남은 ${left}묶음 (${formatDuration(stageElapsed)}째)`
        : left === 0
          ? '마무리하는 중'
          : `답을 기다리는 중 (${formatDuration(stageElapsed)}째)`
      : tr && tr.total
        ? `받은 묶음 ${tr.done} / ${tr.total}`
        : ''
  const steps = retypeset
    ? [{ key: 'typesetting', title: '글자만 다시 쓰기', sub: `${job.pages_done} / ${job.pages_total || book.page_count}쪽` }]
    : [
        {
          key: 'reading',
          title: '읽기',
          sub: job.stage === 'preparing'
            ? dl
              ? `모델 파일 받는 중 (${dl.index}/${dl.count})`
              : '모델 준비 중'
            : `${job.substage || '글자 찾기'}${job.batches > 1 ? ` (묶음 ${job.batch}/${job.batches})` : ''}`,
        },
        { key: 'translating', title: '번역', sub: translateSub },
        { key: 'typesetting', title: '지우고 쓰기', sub: `${job.pages_done} / ${job.pages_total || book.page_count}쪽` },
      ]

  const typesetting = job.stage === 'typesetting'
  const measured = typesetting || !!dl // 진행 막대를 채울 수 있는 때: 쪽 쓰기, 모델 파일 받기
  const percent = typesetting
    ? job.pages_total ? Math.round((job.pages_done / job.pages_total) * 100) : 0
    : dl ? Math.round((dl.done / Math.max(1, dl.total)) * 100) : 0
  const hint =
    job.stage === 'preparing'
      ? dl
        ? '처음 한 번만 모델 파일을 받아요. 받은 파일은 data/models 폴더에 두고 다음부터 그대로 써요.'
        : '모델을 준비하고 있어요. 처음 한 번은 필요한 파일을 받느라 시간이 걸릴 수 있어요.'
      : job.stage === 'translating'
        ? tr && tr.total
          ? '쪽은 다 읽었어요. 남은 번역을 받는 대로 지우고 쓰기로 넘어가요.'
          : '번역 답을 기다리고 있어요. 보통 1~2분 걸려요.'
        : translatingAlongside
          ? '읽은 쪽부터 바로 번역을 맡기고, 그동안 다음 쪽을 읽어요.'
          : typesetting
          ? retypeset
            ? '저장된 번역으로 쪽마다 글자를 다시 써요. 끝난 쪽은 아래에서 바로 볼 수 있어요.'
            : job.pages_skipped
              ? `전에 끝낸 ${job.pages_skipped}쪽은 그대로 두고 남은 쪽을 써요. 끝난 쪽은 아래에서 바로 볼 수 있어요.`
              : '쪽마다 원래 글자를 지우고 한국어를 써요. 끝난 쪽은 아래에서 바로 볼 수 있어요.'
          : '말풍선과 글자를 찾고 일본어를 읽고 있어요.'

  return (
    <section className="panel" aria-label="진행 상황">
      <p className="activity">
        <span className="pulse" aria-hidden="true" />
        <span>{job.stopping ? '멈추는 중이에요' : job.activity}</span>
      </p>
      <div className="steps">
        {steps.map((step, i) => {
          const position = retypeset ? 2 : i
          let state = position < stageIndex ? 'done' : position === stageIndex ? 'current' : 'todo'
          if (position === 1 && translatingAlongside) state = 'current' // 읽는 동안 뒤에서 번역도 돈다
          return (
            <div key={step.key} style={{ display: 'contents' }}>
              {i > 0 && <div className={`step-line${state !== 'todo' ? ' done' : ''}`} />}
              <div className={`step ${state}`}>
                <span className="dot">{state === 'done' ? <CheckIcon size={16} /> : i + 1}</span>
                <span className="label">
                  <b>{step.title}</b>
                  <span className="num">{state === 'todo' ? '기다리는 중' : state === 'done' ? '끝났어요' : step.sub}</span>
                </span>
              </div>
            </div>
          )
        })}
      </div>
      <div
        className={`bar${measured ? '' : ' busy'}`}
        role="progressbar"
        aria-label="진행"
        aria-valuemin={0}
        aria-valuemax={100}
        aria-valuenow={measured ? percent : undefined}
      >
        <span style={measured ? { width: `${percent}%` } : undefined} />
      </div>
      <div className="progress-foot">
        <span>{job.stopping ? stoppingLine(job) : hint}</span>
        <span className="right num">
          {typesetting && eta !== null ? `약 ${formatDuration(eta)} 남음` : `${formatDuration(elapsed)}째`}
        </span>
      </div>
      <Notices job={job} />
    </section>
  )
}

function ResultNotice({ job, book }: { job: Job; book: Book }) {
  let tone = 'info'
  let text = job.message
  if (job.status === 'done') {
    text = job.pages.length
      ? `고친 ${job.pages.length}쪽을 다시 썼어요.`
      : job.mode === 'retypeset'
        ? '글자를 다시 썼어요.'
        : '번역을 모두 끝냈어요.'
  } else if (job.status === 'stopped') {
    text = '멈췄어요. ‘이어서 번역’을 누르면 멈춘 곳부터 다시 해요.'
  } else if (job.status === 'incomplete') {
    tone = 'warning'
    text = incompleteLine(job, book)
  } else if (job.status === 'error') {
    tone = 'error'
  }
  // 이번에 새로 쓴 쪽과 걸린 시간 (단계별 시간 같은 개발용 기록은 '자세한 기록'에)
  const written = Math.max(0, job.pages_done - job.pages_skipped)
  const recap =
    written && !job.pages.length
      ? `이번에 ${written}쪽을 ${formatDuration(job.elapsed_sec)} 만에 끝냈어요.` +
        (job.pages_skipped ? ` 전에 끝낸 ${job.pages_skipped}쪽은 그대로 뒀어요.` : '')
      : ''
  return (
    <section className="panel" aria-label="지난 작업 결과" style={{ gap: 12 }}>
      <div className={`notice ${tone}`} role={tone === 'error' ? 'alert' : 'status'}>
        {tone === 'info' ? <CheckIcon /> : <AlertIcon />}
        <span className="grow">{text}</span>
        {job.error_kind === 'translation_unavailable' && (
          <a className="btn small" href="#/settings">
            연결 확인하기
          </a>
        )}
      </div>
      {recap && <p className="num" style={{ fontSize: 13, color: 'var(--tone)' }}>{recap}</p>}
      <Notices job={job} />
    </section>
  )
}

/** 끝내지 못한 작업의 결과 문구 — 무엇이 남았고 어느 단추로 마저 하는지. */
function incompleteLine(job: Job, book: Book): string {
  if (job.pages.length) {
    // 고른 쪽만 다시 쓴 경우 — 저장까지 끝난 쪽만 done_pages에 들어온다
    const written = new Set(job.done_pages.map((p) => p.name))
    const failed = job.pages.filter((name) => !written.has(name))
    if (failed.length) {
      const pages = book.pages ?? []
      const names = pages.map((p) => p.name)
      const numbers = failed.map((name) => names.indexOf(name) + 1).filter((n) => n > 0).sort((a, b) => a - b)
      const which = !numbers.length
        ? `고친 쪽 가운데 ${failed.length}쪽을`
        : numbers.length > 5
          ? `${numbers.slice(0, 5).join(', ')}쪽 등 ${numbers.length}쪽을`
          : `${numbers.join(', ')}쪽을`
      // '이어서 번역'은 결과가 없는 쪽을 모두 만들고 못 받은 번역도 다시 요청한다 — 다른 곳도 남았으면 함께 한다고 알린다
      const othersLeft = book.failed_lines > 0 || pages.some((p) => !p.typeset && !failed.includes(p.name))
      return `${which} 다시 쓰지 못했어요. ‘이어서 번역’을 누르면 ${othersLeft ? '끝나지 않은 다른 쪽과 함께 ' : ''}다시 만들어요.`
    }
  } else if (book.failed_lines) {
    return `번역을 받지 못한 대사가 ${book.failed_lines}개 남았어요. ‘이어서 번역’을 누르면 그 대사만 다시 요청해요.`
  } else {
    const left = (book.pages ?? []).filter((p) => !p.typeset).length
    if (left) return `아직 만들지 못한 쪽이 ${left}쪽 남았어요. ‘이어서 번역’을 누르면 남은 쪽만 마저 해요.`
  }
  return job.message || '끝나지 않은 쪽이 있어요. ‘이어서 번역’으로 남은 쪽을 마저 해요.'
}

function Notices({ job }: { job: Job }) {
  const logRef = useRef<HTMLPreElement>(null)
  const warnings = job.notices.slice(-3)
  return (
    <>
      {warnings.length > 0 && (
        <div className="notices">
          {warnings.map((n, i) => (
            <div key={i} className={`notice ${n.level === 'error' ? 'error' : 'warning'}`}>
              <AlertIcon size={16} />
              <span className="grow">{n.message}</span>
            </div>
          ))}
        </div>
      )}
      {job.log.length > 0 && (
        <details className="log" onToggle={() => logRef.current?.scrollTo(0, logRef.current.scrollHeight)}>
          <summary>자세한 기록</summary>
          <pre ref={logRef}>{job.log.join('\n')}</pre>
        </details>
      )}
    </>
  )
}

