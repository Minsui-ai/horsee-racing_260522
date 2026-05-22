import re

with open('app.py', 'r', encoding='utf-8') as f:
    content = f.read()

# 1. Header Title
content = content.replace(
    '경마 예측 <span style="font-size: 11px; background-color: #DC2626; color: white; padding: 2px 6px; border-radius: 4px; font-family: monospace; vertical-align: middle; margin-left: 6px;">LIVE</span>',
    '🏇 경마 예측 <span style="font-size: 11px; background-color: #DC2626; color: white; padding: 2px 6px; border-radius: 4px; font-family: monospace; vertical-align: middle; margin-left: 6px;">LIVE</span>'
)

# 2. GateTag Red Badge
content = content.replace(
    'badge_style = "background-color: rgba(69, 10, 10, 0.6); color: #EF4444; border: 1px solid rgba(153, 27, 27, 0.4);"',
    'badge_style = "background-color: rgba(69, 10, 10, 0.6); color: #F87171; border: 1px solid rgba(153, 27, 27, 0.4);"'
)

# 3. EVBadge styling
ev_old = '''    else:
        badge_style = "background-color: rgba(220, 38, 38, 0.15); color: #F87171; border: 1px solid rgba(220, 38, 38, 0.3);"
        sign = ""
    return f"""
    <span class="ev-badge" style="font-family: monospace; font-size: 11px; font-weight: bold; padding: 2px 6px; border-radius: 4px; {badge_style}">{sign}{value:.2f}</span>
    """'''

ev_new = '''    else:
        badge_style = "background-color: rgba(69, 10, 10, 0.8); color: #F87171; border: 1px solid rgba(153, 27, 27, 0.4);"
        sign = ""
    return f"""
    <span class="ev-badge" style="font-family: monospace; font-size: 11px; font-weight: bold; padding: 2px 6px; border-radius: 4px; {badge_style}">{sign}{value:.2f}</span>
    """'''
content = content.replace(ev_old, ev_new)

# 4. bet_budget session state handling
budget_state = '''
# bet_budget 초기화 (View B에서 예상수익 연산을 위해 상단 선언)
if "budget_slider" not in st.session_state:
    st.session_state.budget_slider = 50000
bet_budget = st.session_state.budget_slider
'''
content = content.replace('tabs = st.tabs(["📊 View A: 전체 분석", "🎲 View B: 베팅 결정", "💼 View C: 포트폴리오"])', budget_state + '\ntabs = st.tabs(["📊 View A: 전체 분석", "🎲 View B: 베팅 결정", "💼 View C: 포트폴리오"])')

# 5. View B StrategyCard Mini ProbBar
strat_a_old = '''                    <div style="display: flex; align-items: center; gap: 8px;">
                        <span style="color: #60A5FA; font-weight: bold; font-family: monospace; font-size: 12px;">{h['prob']:.1f}%</span>
                        {get_ev_badge_html(h['ev'])}
                    </div>'''
strat_a_new = '''                    <div style="display: flex; align-items: center; gap: 8px;">
                        {get_prob_bar_html(h['prob'], width_px=48)}
                        {get_ev_badge_html(h['ev'])}
                    </div>'''
content = content.replace(strat_a_old, strat_a_new)

strat_b_old = '''                    <div style="display: flex; align-items: center; gap: 8px;">
                        <span style="color: #34D399; font-weight: bold; font-family: monospace; font-size: 12px;">{h['prob']:.1f}%</span>
                        {get_ev_badge_html(h['ev'])}
                    </div>'''
content = content.replace(strat_b_old, strat_a_new)

strat_c_old = '''                    <div style="display: flex; align-items: center; gap: 8px;">
                        <span style="color: #EF9F27; font-weight: bold; font-family: monospace; font-size: 12px;">{h['prob']:.1f}%</span>
                        {get_ev_badge_html(h['ev'])}
                    </div>'''
content = content.replace(strat_c_old, strat_a_new)

# 6. View B 예상수익 (KRW) 
profit_old = '''        # 예상 수익: EV 양수면 플러스, 음수면 마이너스
        exp_return_val = (h["prob"] / 100) * h["odds"]
        if h["ev"] > 0:
            exp_return_str = f"<span style='color: #34D399; font-weight: bold;'>+{exp_return_val:.2f}배</span>"
        else:
            exp_return_str = f"<span style='color: #F87171; font-weight: bold;'>-{exp_return_val:.2f}배</span>"'''

profit_new = '''        # 예상 수익: EV * 베팅금액
        exp_return_val = h["ev"] * bet_budget
        if h["ev"] > 0:
            exp_return_str = f"<span style='color: #34D399; font-weight: bold;'>+{int(exp_return_val):,}원</span>"
        else:
            exp_return_str = f"<span style='color: #F87171; font-weight: bold;'>{int(exp_return_val):,}원</span>"'''
content = content.replace(profit_old, profit_new)

# 7. View C 안정형 손익분기배당
stable_old = '''                    <div style="display: flex; justify-content: space-between; color: #94A3B8; font-size: 11px; margin-top: 6px;">
                        <span>배분액: {int(alloc_stable):,}원</span>
                        <span style="color: #34D399; font-weight: bold; font-family: monospace;">적중시: {int(alloc_stable * h['odds']):,}원</span>
                    </div>
                </div>'''
stable_new = '''                    <div style="display: flex; justify-content: space-between; color: #94A3B8; font-size: 11px; margin-top: 6px;">
                        <span>배분액: {int(alloc_stable):,}원</span>
                        <span style="color: #34D399; font-weight: bold; font-family: monospace;">적중시: {int(alloc_stable * h['odds']):,}원</span>
                    </div>
                    <div style="text-align: right; color: #64748B; font-size: 10px; margin-top: 4px;">
                        손익분기 배당: {(100 / h['prob']):.1f}배
                    </div>
                </div>'''
content = content.replace(stable_old, stable_new)

# 8. View C 다크호스 최대수익
dh_old = '''                    <div>
                        {dh_body}
                    </div>
                </div>
                <div style="margin-top: 12px; padding-top: 12px; border-top: 1px solid rgba(65, 90, 119, 0.3); font-size: 10px; color: #94A3B8;">
                    주의: 고배당 베팅은 환수율 분산으로 소액 투자 권장
                </div>'''
dh_new = '''                    <div>
                        {dh_body}
                    </div>
                </div>
                <div style="margin-top: 12px; padding-top: 12px; border-top: 1px solid rgba(65, 90, 119, 0.3); font-size: 10px; color: #94A3B8;">
                    <div style="color: #60A5FA; font-weight: bold; font-size: 11px; margin-bottom: 4px;">
                        예상 최대수익: {int(max([(bet_budget / len(dh_port_horses)) * hx['odds'] for hx in dh_port_horses]) - bet_budget) if len(dh_port_horses) > 0 else 0:,}원
                    </div>
                    주의: 고배당 베팅은 환수율 분산으로 소액 투자 권장
                </div>'''
content = content.replace(dh_old, dh_new)

# 9. View C 켈리 테이블
kelly_old = '''            for ka in kelly_allocations[:3]:
                if ka["amount"] > 0:
                    kelly_body += f"""
                    <div style="background-color: rgba(6, 11, 19, 0.6); padding: 10px; border-radius: 8px; border: 1px solid #1B263B; margin-bottom: 8px; font-size: 12px;">
                        <div style="display: flex; justify-content: space-between; font-weight: bold; color: #F8FAFC;">
                            <span>{ka['name']}</span>
                            <span style="color: #EF9F27; font-family: monospace;">{ka['ratio']*100:.0f}% 비율</span>
                        </div>
                        <div style="display: flex; justify-content: space-between; color: #94A3B8; font-size: 11px; margin-top: 6px;">
                            <span>권장 베팅: {ka['amount']:,}원</span>
                            <span style="color: #F5C842; font-weight: bold; font-family: monospace;">적중시: {int(ka['amount'] * ka['odds']):,}원</span>
                        </div>
                    </div>
                    """'''
kelly_new = '''            kelly_body += """
            <div style="overflow-x: auto;">
                <table style="width: 100%; text-align: left; font-size: 12px; border-collapse: collapse;">
                    <thead>
                        <tr style="color: #64748B; border-bottom: 1px solid #1B263B; font-size: 10px; font-weight: bold;">
                            <th style="padding-bottom: 8px;">마필명</th>
                            <th style="padding-bottom: 8px; text-align: center;">비율</th>
                            <th style="padding-bottom: 8px; text-align: right;">권장배팅</th>
                            <th style="padding-bottom: 8px; text-align: right;">적중예상</th>
                        </tr>
                    </thead>
                    <tbody style="border-top: 1px solid rgba(27,38,59,0.3);">
            """
            for ka in kelly_allocations[:3]:
                if ka["amount"] > 0:
                    kelly_body += f"""
                        <tr style="border-bottom: 1px solid rgba(27,38,59,0.2);">
                            <td style="padding: 10px 0; font-weight: bold; color: #E2E8F0;">{ka['name']}</td>
                            <td style="padding: 10px 0; text-align: center; font-family: monospace; color: #EF9F27; font-weight: bold;">{ka['ratio']*100:.0f}%</td>
                            <td style="padding: 10px 0; text-align: right; font-family: monospace; color: #CBD5E1;">{ka['amount']:,}원</td>
                            <td style="padding: 10px 0; text-align: right; font-family: monospace; font-weight: bold; color: #F5C842;">{int(ka['amount'] * ka['odds']):,}원</td>
                        </tr>
                    """
            kelly_body += """
                    </tbody>
                </table>
            </div>
            """'''
content = content.replace(kelly_old, kelly_new)

# 10. View C 기대수익 테이블 손익분기 컬럼
table_header_old = '''                            <th style="padding: 8px 0; font-size: 10px; font-weight: bold; border: none; color: #778DA9; text-align: right;">기대 수익 (EV)</th>
                        </tr>
                    </thead>'''
table_header_new = '''                            <th style="padding: 8px 0; font-size: 10px; font-weight: bold; border: none; color: #778DA9; text-align: right;">손익분기</th>
                            <th style="padding: 8px 0; font-size: 10px; font-weight: bold; border: none; color: #778DA9; text-align: right;">기대 수익 (EV)</th>
                        </tr>
                    </thead>'''
content = content.replace(table_header_old, table_header_new)

# Add Break Even variables and highlighted rows
# Win row
win_old = '''                        <tr style="border-bottom: 1px solid rgba(27,38,59,0.2);">
                            <td style="padding: 10px 0; font-weight: bold;">단승 (Win)</td>
                            <td style="padding: 10px 0; color: #CBD5E1;">{h0['name']}</td>
                            <td style="padding: 10px 0; text-align: right; font-family: monospace; font-weight: bold; color: #F5C842;">{odds_win:.1f}배</td>
                            <td style="padding: 10px 0; text-align: right; font-family: monospace;">{prob_win:.1f}%</td>
                            <td style="padding: 10px 0; text-align: right; font-family: monospace; font-weight: bold; color: {ev_win_color};">{ev_win_sign}{int(ev_win_val):,}원</td>
                        </tr>'''
win_new = '''                        <tr style="border-bottom: 1px solid rgba(27,38,59,0.2); {'background-color: rgba(16, 185, 129, 0.1); border-left: 4px solid #10B981;' if ev_win_val > 0 else ''}">
                            <td style="padding: 10px 0; font-weight: bold; padding-left: 8px;">단승 (Win)</td>
                            <td style="padding: 10px 0; color: #CBD5E1;">{h0['name']}</td>
                            <td style="padding: 10px 0; text-align: right; font-family: monospace; font-weight: bold; color: #F5C842;">{odds_win:.1f}배</td>
                            <td style="padding: 10px 0; text-align: right; font-family: monospace;">{prob_win:.1f}%</td>
                            <td style="padding: 10px 0; text-align: right; font-family: monospace; color: #64748B;">{(100 / (prob_win or 1)):.1f}배</td>
                            <td style="padding: 10px 0; text-align: right; font-family: monospace; font-weight: bold; color: {ev_win_color};">{ev_win_sign}{int(ev_win_val):,}원</td>
                        </tr>'''
content = content.replace(win_old, win_new)

# Place row
place_old = '''                        <tr style="border-bottom: 1px solid rgba(27,38,59,0.2);">
                            <td style="padding: 10px 0; font-weight: bold;">연승 (Place)</td>
                            <td style="padding: 10px 0; color: #CBD5E1;">{h0['name']}</td>
                            <td style="padding: 10px 0; text-align: right; font-family: monospace; font-weight: bold; color: #F5C842;">{odds_place:.1f}배</td>
                            <td style="padding: 10px 0; text-align: right; font-family: monospace;">{prob_place:.1f}%</td>
                            <td style="padding: 10px 0; text-align: right; font-family: monospace; font-weight: bold; color: {ev_place_color};">{ev_place_sign}{int(ev_place_val):,}원</td>
                        </tr>'''
place_new = '''                        <tr style="border-bottom: 1px solid rgba(27,38,59,0.2); {'background-color: rgba(16, 185, 129, 0.1); border-left: 4px solid #10B981;' if ev_place_val > 0 else ''}">
                            <td style="padding: 10px 0; font-weight: bold; padding-left: 8px;">연승 (Place)</td>
                            <td style="padding: 10px 0; color: #CBD5E1;">{h0['name']}</td>
                            <td style="padding: 10px 0; text-align: right; font-family: monospace; font-weight: bold; color: #F5C842;">{odds_place:.1f}배</td>
                            <td style="padding: 10px 0; text-align: right; font-family: monospace;">{prob_place:.1f}%</td>
                            <td style="padding: 10px 0; text-align: right; font-family: monospace; color: #64748B;">{(100 / (prob_place or 1)):.1f}배</td>
                            <td style="padding: 10px 0; text-align: right; font-family: monospace; font-weight: bold; color: {ev_place_color};">{ev_place_sign}{int(ev_place_val):,}원</td>
                        </tr>'''
content = content.replace(place_old, place_new)

# Quinella row
quin_old = '''                        <tr>
                            <td style="padding: 10px 0; font-weight: bold;">복승 (Quinella)</td>
                            <td style="padding: 10px 0; color: #CBD5E1;">{h0['name']} + {h1['name']}</td>
                            <td style="padding: 10px 0; text-align: right; font-family: monospace; font-weight: bold; color: #F5C842;">{odds_quin:.1f}배</td>
                            <td style="padding: 10px 0; text-align: right; font-family: monospace;">{prob_quin:.1f}%</td>
                            <td style="padding: 10px 0; text-align: right; font-family: monospace; font-weight: bold; color: {ev_quin_color};">{ev_quin_sign}{int(ev_quin_val):,}원</td>
                        </tr>'''
quin_new = '''                        <tr style="{'background-color: rgba(16, 185, 129, 0.1); border-left: 4px solid #10B981;' if ev_quin_val > 0 else ''}">
                            <td style="padding: 10px 0; font-weight: bold; padding-left: 8px;">복승 (Quinella)</td>
                            <td style="padding: 10px 0; color: #CBD5E1;">{h0['name']} + {h1['name']}</td>
                            <td style="padding: 10px 0; text-align: right; font-family: monospace; font-weight: bold; color: #F5C842;">{odds_quin:.1f}배</td>
                            <td style="padding: 10px 0; text-align: right; font-family: monospace;">{prob_quin:.1f}%</td>
                            <td style="padding: 10px 0; text-align: right; font-family: monospace; color: #64748B;">{(100 / (prob_quin or 1)):.1f}배</td>
                            <td style="padding: 10px 0; text-align: right; font-family: monospace; font-weight: bold; color: {ev_quin_color};">{ev_quin_sign}{int(ev_quin_val):,}원</td>
                        </tr>'''
content = content.replace(quin_old, quin_new)

# 11. View C 손익분기 역산 메시지
msg_old = '''                <div style="display: flex; justify-content: space-between; font-size: 12px; margin-top: 4px;">
                    <span style="color: #94A3B8;">현재 시장 배당률</span>
                    <span style="font-weight: 900; color: #F5C842;">{bh['odds']:.1f}배</span>
                </div>'''
msg_new = '''                <div style="font-size: 12px; color: #E2E8F0; font-weight: bold; margin-bottom: 4px;">
                    📢 이 말이 이기려면 배당률이 최소 {break_even_odds:.1f}배 이상이어야 합니다. (현재 시장 배당: {bh['odds']:.1f}배)
                </div>
                <div style="display: flex; justify-content: space-between; font-size: 12px; margin-top: 4px;">
                    <span style="color: #94A3B8;">현재 시장 배당률</span>
                    <span style="font-weight: 900; color: #F5C842;">{bh['odds']:.1f}배</span>
                </div>'''
content = content.replace(msg_old, msg_new)

with open('app.py', 'w', encoding='utf-8') as f:
    f.write(content)

print("Modification complete.")
