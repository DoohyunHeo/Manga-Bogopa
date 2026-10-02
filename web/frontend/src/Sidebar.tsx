import { useState } from 'react'
import { pageUrl, type Book, type Job } from './api'
import { useApp } from './App'
import { AlertIcon, CheckIcon, PlusIcon, SettingsIcon } from './icons'

/** 책장 목록과 첫 화면에서 쓰는 한 줄 상태. */
export function statusLine(book: Book, job: Job | null): { text: string; tone: '' | 'active' | 'attention' } {
  if (job && job.book_id === book.id && job.status === 'running') {
    if (job.stopping) return { text: '멈추는 중', tone: 'active' }
    if (job.stage === 'preparing') return { text: '준비하는 중', tone: 'active' }
    if (job.stage === 'typesetting')
      return { text: job.pages_total ? `${job.pages_done} / ${job.pages_total}쪽 쓰는 중` : '쓰는 중', tone: 'active' }
    if (job.stage === 'translating') return { text: '번역하는 중', tone: 'active' }
    return { text: '읽는 중', tone: 'active' }
  }
  switch (book.status) {
    case 'done':
      return { text: `${book.page_count}쪽 완료`, tone: '' }
    case 'needs_retry':
      return { text: `번역 못 한 대사 ${book.failed_lines}개`, tone: 'attention' }
    case 'partial':
      return { text: `${book.typeset_count} / ${book.page_count}쪽`, tone: '' }
    case 'missing':
      return { text: '원고 폴더를 찾지 못했어요', tone: 'attention' }
    default:
      return { text: `시작 전 (${book.page_count}쪽)`, tone: '' }
  }
}

export function Cover({ book, width = 120 }: { book: Book; width?: number }) {
  if (!book.cover) return null
  const kind = book.typeset_count > 0 ? 'translated' : 'original'
  return <img src={pageUrl(book.id, book.cover, kind, width, book.typeset_count)} alt="" loading="lazy" />
}

export function Sidebar({ route }: { route: string[] }) {
  const { books, booksLoaded, job, system, pickAndOpen } = useApp()
  const [picking, setPicking] = useState(false)

  const newBook = async () => {
    if (picking) return
    setPicking(true)
    await pickAndOpen()
    setPicking(false)
  }

  return (
    <aside className="sidebar">
      <a className="brand" href="#/">
        만화보고파
      </a>
      <button className="btn primary block" onClick={newBook} aria-disabled={picking}>
        {picking ? <span className="spinner" aria-hidden="true" /> : <PlusIcon />}
        새 책 번역하기
      </button>
      <nav aria-label="책장" style={{ display: 'flex', flexDirection: 'column', gap: 6, minHeight: 0, flex: 1 }}>
        <div className="nav-label">책장</div>
        <div className="book-list">
          {books.map((book) => {
            const line = statusLine(book, job)
            const current = route[0] === 'book' && route[1] === book.id
            return (
              <a key={book.id} className="book-row" href={`#/book/${book.id}`} aria-current={current ? 'page' : undefined}>
                <div className="cover">
                  <Cover book={book} width={80} />
                </div>
                <div className="meta">
                  <span className="title">{book.title}</span>
                  <span className={`status num ${line.tone}`}>{line.text}</span>
                </div>
              </a>
            )
          })}
          {booksLoaded && books.length === 0 && <p style={{ padding: '0 8px', fontSize: 13, color: 'var(--tone)' }}>아직 번역한 책이 없어요.</p>}
        </div>
      </nav>
      <div className="sidebar-foot">
        <a className="nav-row" href="#/settings" aria-current={route[0] === 'settings' ? 'page' : undefined}>
          <SettingsIcon />
          설정
        </a>
        {system && (
          <div className="connection">
            {system.translation_found ? (
              <span className="ok">
                <CheckIcon size={16} />
              </span>
            ) : (
              <span className="bad">
                <AlertIcon size={16} />
              </span>
            )}
            {system.translation_found ? '번역 연결됨' : '번역 연결 필요'}
          </div>
        )}
      </div>
    </aside>
  )
}
