import { useEffect, useState, type RefObject } from 'react'
import type { Job } from './api'

// 창 제목 = [작업 진행] + 지금 화면 + 이름 — 다른 창을 보고 있어도 탭·작업 표시줄에서 진행이 보이게
let titleBase = ''
let titleProgress = ''
const renderTitle = () => {
  document.title = `${titleProgress}${titleBase ? `${titleBase} — ` : ''}만화보고파`
}

/** 창 제목을 지금 화면에 맞추고, 화면이 닫히면 앞 제목으로 돌려놓는다 (쪽 보기를 닫으면 책 제목으로). */
export function useDocumentTitle(title: string) {
  useEffect(() => {
    const previous = titleBase
    titleBase = title
    renderTitle()
    return () => {
      titleBase = previous
      renderTitle()
    }
  }, [title])
}

/** 창 제목 앞에 작업 진행을 붙인다 (빈 글자면 뗀다). */
export function setTitleProgress(prefix: string) {
  titleProgress = prefix
  renderTitle()
}

const FOCUSABLE = 'a[href], button:not([disabled]), input:not([disabled]), select:not([disabled]), textarea:not([disabled]), [tabindex]:not([tabindex="-1"])'

/** 창(대화 상자)이 열려 있는 동안 Tab 초점을 그 안에 가두고, 닫히면 연 자리로 돌려놓는다. */
export function useFocusTrap(ref: RefObject<HTMLElement | null>, active = true) {
  // 닫힐 때만 연 자리로 돌려놓는다 (안에서 확인 창이 잠시 초점을 가져가도 그대로)
  useEffect(() => {
    const opener = document.activeElement as HTMLElement | null
    return () => {
      if (opener && document.contains(opener)) opener.focus()
    }
  }, [])
  useEffect(() => {
    if (!active) return
    const onKey = (e: KeyboardEvent) => {
      if (e.key !== 'Tab' || !ref.current) return
      const items = Array.from(ref.current.querySelectorAll<HTMLElement>(FOCUSABLE)).filter((el) => el.offsetParent !== null)
      if (!items.length) return
      const first = items[0]
      const last = items[items.length - 1]
      if (e.shiftKey && document.activeElement === first) {
        e.preventDefault()
        last.focus()
      } else if (!e.shiftKey && document.activeElement === last) {
        e.preventDefault()
        first.focus()
      } else if (!ref.current.contains(document.activeElement)) {
        e.preventDefault()
        first.focus()
      }
    }
    document.addEventListener('keydown', onKey)
    return () => document.removeEventListener('keydown', onKey)
  }, [ref, active])
}

/** 주소창의 #/… 경로 — 뒤로 가기·새로 고침에도 같은 화면이 열린다. */
export function useRoute(): string[] {
  const [hash, setHash] = useState(window.location.hash)
  useEffect(() => {
    const onChange = () => setHash(window.location.hash)
    window.addEventListener('hashchange', onChange)
    return () => window.removeEventListener('hashchange', onChange)
  }, [])
  return hash.replace(/^#\/?/, '').split('/').filter(Boolean).map(decodeURIComponent)
}

export const go = (path: string) => {
  window.location.hash = `#/${path}`
}

/** 서버가 흘려보내는 작업 상태 (SSE)와 연결 여부. 끊기면 브라우저가 알아서 다시 잇는다. */
export function useJob(): [Job | null, boolean] {
  const [job, setJob] = useState<Job | null>(null)
  const [connected, setConnected] = useState(true)
  useEffect(() => {
    const source = new EventSource('/api/job/events')
    source.onopen = () => setConnected(true)
    source.onerror = () => setConnected(false)
    source.onmessage = (event) => {
      setConnected(true)
      const data = JSON.parse(event.data)
      setJob(data && data.book_id ? ({ ...data, received_at: Date.now() } as Job) : null)
    }
    return () => source.close()
  }, [])
  return [job, connected]
}

/** 1초마다 바뀌는 지금 시각 — 새 알림 사이에도 걸린 시간을 올리려고. */
export function useNow(active: boolean): number {
  const [now, setNow] = useState(Date.now())
  useEffect(() => {
    if (!active) return
    const timer = window.setInterval(() => setNow(Date.now()), 1000)
    return () => window.clearInterval(timer)
  }, [active])
  return now
}

/** 걸린·남은 시간을 말로. 10분 안쪽은 초까지 — 기다리는 동안 숫자가 멈춰 보이지 않게. */
export function formatDuration(seconds: number): string {
  const total = Math.max(1, Math.round(seconds))
  if (total < 60) return `${total}초`
  if (total < 600) {
    const rest = total % 60
    return `${Math.floor(total / 60)}분${rest ? ` ${rest}초` : ''}`
  }
  const minutes = Math.round(total / 60)
  if (minutes < 60) return `${minutes}분`
  return `${Math.floor(minutes / 60)}시간 ${minutes % 60}분`
}
