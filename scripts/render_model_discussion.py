"""Render the model discussion as a standalone Korean HTML document."""

if __name__ == "__main__":
    from _execution_bootstrap import launch
    launch(__file__)

from html import escape
from pathlib import Path
import re

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / 'docs/research/model-comparison.md'
OUTPUT = SOURCE.with_suffix('.html')


def inline(value):
    value = escape(value)
    value = re.sub(r'\[([^\]]+)\]\(([^)]+)\)', r'<a href="\2">\1</a>', value)
    value = re.sub(r'\*\*(.+?)\*\*', r'<strong>\1</strong>', value)
    return re.sub(r'`([^`]+)`', r'<code>\1</code>', value)


def render(text):
    lines = text.splitlines()
    sections, nav, paragraph = [], [], []
    opened, index = False, 0
    def flush():
        if paragraph:
            sections.append('<p>' + inline(' '.join(paragraph)) + '</p>')
            paragraph.clear()
    i = 0
    while i < len(lines):
        line = lines[i].strip()
        if line.startswith('# '):
            i += 1
            continue
        if line.startswith('## '):
            flush()
            if opened:
                sections.append('</section>')
            index += 1
            title = line[3:]
            nav.append(f'<a href="#section-{index}"><span>0{index}</span>{escape(title)}</a>')
            sections.append(f'<section id="section-{index}"><div class="eyebrow">DISCUSSION / 0{index}</div><h2>{escape(title)}</h2>')
            opened = True
        elif line.startswith('|'):
            flush()
            rows = []
            while i < len(lines) and lines[i].strip().startswith('|'):
                cells = [cell.strip() for cell in lines[i].strip().strip('|').split('|')]
                if not all(re.fullmatch(r'[-: ]+', cell) for cell in cells):
                    rows.append(cells)
                i += 1
            sections.append('<div class="table-tools"><span>같은 목표, 다른 학습 방식</span><button id="filter" type="button" aria-pressed="false">우선 검토 후보만 보기</button></div><div class="table-scroll" tabindex="0" role="region" aria-label="학습 모델 비교표"><table><caption class="sr-only">모델별 강점과 적용 한계</caption><thead><tr>')
            sections.extend('<th scope="col">' + inline(cell) + '</th>' for cell in rows[0])
            sections.append('</tr></thead><tbody>')
            for row in rows[1:]:
                recommended = 'PPO' in row[0]
                candidate = recommended or 'Dreamer' in row[0]
                sections.append(f'<tr data-candidate="{str(candidate).lower()}" class="{"recommended" if recommended else ""}">')
                sections.append('<th scope="row">' + inline(row[0]) + ('<span class="tag">첫 비교 후보</span>' if recommended else '') + '</th>')
                sections.extend('<td>' + inline(cell) + '</td>' for cell in row[1:])
                sections.append('</tr>')
            sections.append('</tbody></table></div>')
            continue
        elif not line:
            flush()
        else:
            paragraph.append(line)
        i += 1
    flush()
    if opened:
        sections.append('</section>')
    return '\n'.join(sections), '\n'.join(nav)


STYLE = '''
:root{--paper:#f5f3ed;--ink:#1d2d35;--muted:#54666c;--line:#d5dcd7;--accent:#17675b;--tint:#e5efe8}
*{box-sizing:border-box}html{scroll-behavior:smooth;scroll-padding-top:30px}body{margin:0;background:var(--paper);color:var(--ink);font-family:"Segoe UI","Malgun Gothic",sans-serif;line-height:1.85;word-break:keep-all}a{color:var(--accent);text-underline-offset:4px}button{font:inherit;cursor:pointer;border:1px solid var(--line);border-radius:5px;padding:7px 14px;background:white;color:var(--ink)}button:hover{background:var(--tint)}button:focus-visible,a:focus-visible,summary:focus-visible,[tabindex]:focus-visible{outline:3px solid #bf7625;outline-offset:4px}.shell{max-width:1440px;margin:auto;display:grid;grid-template-columns:255px minmax(0,1fr);gap:65px;padding:45px 50px 90px}aside{align-self:start;position:sticky;top:35px}.brand{font-weight:800;letter-spacing:2px;font-size:16px}.edition{color:var(--muted);font-size:12px;margin:8px 0 35px}nav a{display:block;text-decoration:none;color:var(--muted);font-size:13px;padding:12px 0;border-bottom:1px solid var(--line)}nav span{display:inline-block;width:27px;font-size:11px;color:var(--accent)}.aside-note{margin-top:28px;font-size:12px;color:var(--muted)}main{min-width:0}.eyebrow{font-size:11px;letter-spacing:2px;font-weight:700;color:var(--accent)}header{padding:0 0 32px;border-bottom:2px solid var(--ink)}h1{font-size:clamp(32px,3.8vw,53px);line-height:1.3;letter-spacing:-2px;margin:20px 0}h2{font-size:27px;line-height:1.45;letter-spacing:-1px;margin:8px 0 24px}.lead{font-size:18px;max-width:700px;color:var(--muted)}.status{display:inline-block;border:1px solid var(--accent);padding:3px 10px;font-size:12px;color:var(--accent);border-radius:3px}.hero-note{background:var(--tint);padding:22px 26px;border-left:4px solid var(--accent);margin:28px 0}.hero-note p{margin:4px 0}.flow{display:grid;grid-template-columns:repeat(4,1fr);gap:12px;margin:26px 0}.flow div{padding:18px 14px;background:#fff;border-top:3px solid var(--accent)}.flow b{display:block;font-size:14px}.flow small{display:block;color:var(--muted);font-size:12px;margin-top:6px}.flow em{font-style:normal;color:var(--accent);font-size:11px}section{padding:40px 0 20px;border-top:1px solid var(--line);margin-top:25px}p{font-size:15px;margin:18px 0}strong{color:#123c34}code{font-size:.9em;background:#e6e9e4;padding:2px 5px;border-radius:3px}.table-tools{display:flex;gap:15px;justify-content:space-between;align-items:center;margin-bottom:12px;font-size:12px;color:var(--muted)}.table-tools button{font-size:12px}.table-scroll{overflow-x:auto;border:1px solid var(--line);background:#fff}table{width:100%;min-width:650px;border-collapse:collapse;font-size:13px;line-height:1.8}th,td{text-align:left;vertical-align:top;padding:19px 18px;border-bottom:1px solid var(--line)}thead{background:var(--ink);color:#fff}thead th:first-child{width:20%}tbody th{font-weight:700}tbody tr:last-child>*{border-bottom:0}tr.recommended{background:var(--tint)}.tag{display:block;font-size:10px;color:var(--accent);margin-top:7px}.sr-only{position:absolute;width:1px;height:1px;padding:0;margin:-1px;overflow:hidden;clip:rect(0,0,0,0)}[hidden]{display:none!important}.glossary{margin:26px 0}details{border-bottom:1px solid var(--line);padding:12px 0}summary{cursor:pointer;font-size:14px;font-weight:600}details p{font-size:13px;color:var(--muted);margin:10px 0}footer{border-top:2px solid var(--ink);padding-top:20px;margin-top:45px;color:var(--muted);font-size:12px}.print{margin-top:25px;font-size:12px}
@media(max-width:1000px){.shell{grid-template-columns:190px minmax(0,1fr);gap:30px;padding:30px}.flow{grid-template-columns:repeat(2,1fr)}}
@media(max-width:720px){.shell{display:block;padding:24px 20px}aside{position:static}.edition{margin-bottom:15px}nav{display:flex;gap:8px;overflow:auto;margin-bottom:26px}nav a{flex:0 0 auto;padding:6px;font-size:12px}nav span,.aside-note,.print{display:none}h1{letter-spacing:-1px}.lead{font-size:16px}.table-tools{align-items:flex-start}section{padding-top:28px}.hero-note{padding:18px}p{font-size:14px}}
@media(prefers-reduced-motion:reduce){html{scroll-behavior:auto}}
@media print{body{background:white}.shell{display:block;padding:0;max-width:none}aside,.table-tools button{display:none}.flow{grid-template-columns:repeat(4,1fr)}section{break-inside:auto}.table-scroll{overflow:visible}table{min-width:0}tr{break-inside:avoid}tr[hidden]{display:table-row!important}header{break-after:avoid}h2{break-after:avoid}a{color:inherit}details p{display:block}footer{font-size:10px}}
'''


def main():
    body, nav = render(SOURCE.read_text(encoding='utf-8'))
    page = '''<!doctype html><html lang="ko"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>PlayModel — 전체 게임 학습 모델 논의</title><meta name="description" content="Pluto, AlphaStar, 재귀 PPO, DreamerV3 비교와 로컬 게임 학습 모델 설계 논의"><style>''' + STYLE + '''</style></head><body>
<div class="shell"><aside><div class="brand">PLAYMODEL</div><div class="edition">설계 논의 문서 · 2026.09.27</div><nav aria-label="문서 목차">''' + nav + '''</nav><p class="aside-note">로컬 실행 · 전체 판 학습<br>모델 선택 전 비교와 검증</p><button class="print" type="button" onclick="window.print()">인쇄 / PDF 저장</button></aside>
<main><header><div class="eyebrow">MODEL DESIGN DISCUSSION</div><h1>게임 전체를 배우는<br>모델은 어떻게 만들까?</h1><p class="lead">현재 이동 모델에서 전체 게임 판단으로.<br>Pluto·AlphaStar·PPO·DreamerV3를 같은 목표에서 비교합니다.</p><span class="status">논의안 · 최종 모델 미확정</span></header>
<div class="hero-note"><div class="eyebrow">권장 비교 순서</div><p><strong>공유 화면 인코더 + GRU 기억 + PPO</strong>를 첫 주력 후보로 검토합니다.</p><p>현재 선형 정책을 기준선으로 유지하고, DreamerV3는 후속 비교군으로 둡니다. 구현 완료나 실력 향상을 뜻하지 않습니다.</p></div>
<div class="flow" aria-label="제안하는 전체 판 학습 흐름"><div><em>01 · 관측</em><b>화면과 현재 상태</b><small>체력·장비·후보·관측 신뢰도</small></div><div><em>02 · 기억과 판단</em><b>상황에 맞는 행동</b><small>이동·무기·성장·구매</small></div><div><em>03 · 결과 확인</em><b>실제로 바뀌었는가</b><small>입력 전송과 적용 결과 분리</small></div><div><em>04 · 학습과 평가</em><b>한 판의 경험으로 개선</b><small>별도 판에서 비교 후 모델 교체</small></div></div>
<div class="glossary" aria-label="용어 설명"><details><summary>OCR · 화면의 글자를 읽는 기능</summary><p>이미지의 ‘가격 20’을 문자와 숫자로 바꾸는 기능입니다. 읽었다고 효과를 이해했거나 모델이 학습된 것은 아닙니다.</p></details><details><summary>GRU · 최근 상황을 기억하는 신경망</summary><p>현재 화면만 보는 대신 이전 관측과 행동의 정보를 이어받습니다. 어떤 정보를 잘 기억하는지는 학습과 검증이 필요합니다.</p></details><details><summary>PPO · 행동 결과로 정책을 개선하는 학습 방법</summary><p>직접 행동해서 얻은 경험으로 다음 행동의 확률을 조정합니다. 업데이트가 과하게 변하지 않도록 제어하지만, 성공을 보장하지는 않습니다.</p></details><details><summary>World model · 행동 뒤의 상황을 예측하는 모델</summary><p>경험에서 다음 상태와 보상을 예측하도록 학습합니다. Dreamer는 그 예측 안에서도 행동을 연습합니다. 예측 오류가 실제 판단을 망칠 수 있어 별도 검증이 필요합니다.</p></details></div>
''' + body + '''<footer>원문: <a href="model-comparison.md">model-comparison.md</a> · 외부 서비스 없이 열리는 단일 HTML 문서.<br>출처와 측정 조건은 본문에 보존했습니다. 학습 모델의 성능 달성 보고가 아닙니다.</footer></main></div>
<script>const filter=document.getElementById('filter');filter?.addEventListener('click',()=>{const active=filter.getAttribute('aria-pressed')!=='true';filter.setAttribute('aria-pressed',String(active));filter.textContent=active?'모든 모델 보기':'우선 검토 후보만 보기';document.querySelectorAll('tr[data-candidate]').forEach(row=>row.hidden=active&&row.dataset.candidate!=='true');});window.addEventListener('beforeprint',()=>document.querySelectorAll('details').forEach(d=>{d.dataset.wasOpen=String(d.open);d.open=true;}));window.addEventListener('afterprint',()=>document.querySelectorAll('details').forEach(d=>d.open=d.dataset.wasOpen==='true'));</script></body></html>'''
    if '사용자가 권장 주력 구조' in SOURCE.read_text(encoding='utf-8'):
        page = page.replace('논의안 · 최종 모델 미확정', '주력 구조 채택 · 성능 검증 진행')
        page = page.replace('첫 주력 후보로 검토합니다.', '주력 구조로 채택했습니다.')
    OUTPUT.write_text(page, encoding='utf-8')
    assert page.count('<section id=') == 5
    assert page.count('data-candidate=') == 5
    assert 'src="http' not in page and '<html lang="ko">' in page
    print(OUTPUT)
    print(f'HTML verified: {len(page.encode("utf-8"))} bytes, 5 sections, 5 model rows')


if __name__ == '__main__':
    main()
