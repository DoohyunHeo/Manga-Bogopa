import type { Job } from './api'
import { formatDuration } from './hooks'
import { AlertIcon, CheckIcon, CloseIcon } from './icons'

/** 멈추기를 누른 뒤 무엇을 마치고 멈추는지 — 단계마다 다르다. */
export function stoppingLine(job: Job): string {
  if (job.stage === 'typesetting') return '지금 쓰던 쪽들을 마치고 멈춰요. 끝낸 쪽은 저장돼요.'
  if (job.stage === 'reading' || job.stage === 'translating')
    return '읽던 묶음과 이미 보낸 번역을 마치고 멈춰요. 읽은 쪽은 저장돼서, 이어서 번역하면 번역만 다시 해요.'
  return '곧 멈춰요.'
}

/** 뒤에서 도는 번역 가운데 아직 안 받은 묶음 수 (번역 정보가 없으면 null). */
export function translationLeft(job: Job): number | null {
  return job.translation && job.translation.total ? job.translation.total - job.translation.done : null
}

/** 지금 작업이 어디까지 왔는지 한 줄로 — 책장·설정·다른 책 화면에서도 보인다. */
export function jobLine(job: Job): string {
  if (job.status === 'running') {
    if (job.stopping) return '멈추는 중이에요'
    switch (job.stage) {
      case 'preparing':
        return job.download ? `모델 파일을 받는 중이에요 (${job.download.index}/${job.download.count})` : '준비하는 중이에요'
      case 'reading':
        return `읽는 중이에요${job.batches > 1 ? ` (묶음 ${job.batch}/${job.batches})` : ''}`
      case 'translating': {
        const left = translationLeft(job)
        return left
          ? `번역을 기다리는 중이에요 (남은 ${left}묶음)`
          : `번역을 기다리는 중이에요 (${formatDuration(job.stage_elapsed_sec)}째)`
      }
      case 'typesetting': {
        const eta = job.eta_sec !== null ? `, 약 ${formatDuration(job.eta_sec)} 남음` : ''
        const count = job.pages_total ? ` (${job.pages_done} / ${job.pages_total}쪽${eta})` : ''
        return `${job.mode === 'retypeset' ? '다시 쓰는' : '지우고 쓰는'} 중이에요${count}`
      }
    }
  }
  if (job.status === 'done') {
    if (job.pages.length) return `고친 ${job.pages.length}쪽을 다시 썼어요`
    return job.mode === 'retypeset' ? '글자를 다시 썼어요' : '번역을 모두 끝냈어요'
  }
  if (job.status === 'stopped') return '멈췄어요. 책 화면에서 이어서 번역할 수 있어요'
  if (job.status === 'incomplete') return '끝나지 않은 쪽이 남았어요. 책 화면에서 확인해 주세요'
  return '작업이 멈췄어요. 책 화면에서 이유를 확인해 주세요'
}

export function typesetPercent(job: Job): number | null {
  return job.status === 'running' && job.stage === 'typesetting' && job.pages_total
    ? Math.round((job.pages_done / job.pages_total) * 100)
    : null
}

export function StatusBar({ job, onDismiss }: { job: Job; onDismiss: () => void }) {
  const running = job.status === 'running'
  const percent = typesetPercent(job)
  const tone = running ? 'running' : job.status === 'done' ? 'done' : job.status === 'error' ? 'error' : 'warning'
  return (
    <div className={`statusbar ${tone}`} role="region" aria-label="작업 상태">
      {/* 쪽마다 바뀌는 숫자는 읽어 주지 않고, 끝났을 때만 한 번 알린다 */}
      <span className="sr-only" role="status">
        {running ? '' : `‘${job.title}’ ${jobLine(job)}`}
      </span>
      <span className="icon" aria-hidden="true">
        {running ? <span className="spinner" /> : tone === 'done' ? <CheckIcon /> : <AlertIcon />}
      </span>
      <span className="text">
        <b>{job.title}</b>
        <span className="num">{jobLine(job)}</span>
      </span>
      {percent !== null && (
        <span className="mini-bar" aria-hidden="true">
          <span style={{ width: `${percent}%` }} />
        </span>
      )}
      <a className="btn small" href={`#/book/${job.book_id}`}>
        보기
      </a>
      {!running && (
        <button className="btn small icon" aria-label="알림 닫기" onClick={onDismiss}>
          <CloseIcon size={16} />
        </button>
      )}
    </div>
  )
}
